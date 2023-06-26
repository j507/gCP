#include <gCP/crystal_data.h>

#include <deal.II/base/tensor.h>

#include <fstream>
#include <math.h>



namespace gCP
{



template<int dim>
CrystalsData<dim>::CrystalsData()
:
pcout(std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
flag_init_was_called(false)
{}



template<int dim>
void CrystalsData<dim>::init(
  const dealii::Triangulation<dim>  &triangulation,
  const std::string                 crystal_orientations_file_name,
  const std::string                 slip_directions_file_name,
  const std::string                 slip_normals_file_name)
{
  count_n_crystals(triangulation);

  read_and_store_data(crystal_orientations_file_name,
                      slip_directions_file_name,
                      slip_normals_file_name);

  compute_rotation_matrices();

  compute_3d_rotation_matrices();

  compute_slip_systems();

  flag_init_was_called = true;
}



template<int dim>
void CrystalsData<dim>::count_n_crystals(
  const dealii::Triangulation<dim>  &triangulation)
{
  std::set<dealii::types::material_id> crystal_id_set;

  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      if (!crystal_id_set.count(cell->material_id()))
        crystal_id_set.emplace(cell->material_id());

  crystal_id_set =
    dealii::Utilities::MPI::compute_set_union(crystal_id_set,
                                              MPI_COMM_WORLD);

  n_crystals = crystal_id_set.size();
}



template<int dim>
void CrystalsData<dim>::read_and_store_data(
  const std::string crystal_orientations_file_name,
  const std::string slip_directions_file_name,
  const std::string slip_normals_file_name)
{
  // String indicated the file extension of the input files
  std::string file_extension = ".txt";

  // Lambda function that reads tabular data of input_file and stores
  // it into write_into. The data has to be separated by commas.
  auto read_and_store =
    [&](std::ifstream                       &input_file,
        std::vector<dealii::Tensor<1,dim>>  &write_into,
        const bool                          flag_reading_euler_angles)
    {
      unsigned int component;

      std::string line;

      while(std::getline(input_file, line))
      {
        // Check if the line is empty
        {
          std::string line_with_no_spaces = line;

          line_with_no_spaces.erase(
            remove(line_with_no_spaces.begin(),
                   line_with_no_spaces.end(),
                   ' '),
            line_with_no_spaces.end());

          if (line_with_no_spaces.empty())
            break;
        }

        dealii::Tensor<1,dim> vector;

        std::stringstream     line_as_stream_input(line);

        std::string           vector_component;

        component = 0;

        while(std::getline(line_as_stream_input,
                            vector_component,
                            ','))
        {
          AssertIndexRange(component, dim);

          vector[component++] = std::stod(vector_component);
        }

        if (flag_reading_euler_angles && dim ==2)
        {
          AssertThrow(
            component == 1,
            dealii::ExcMessage(
              ("In 2D only rotations of the xy plane are allowed. "
               "Nonetheless, more than one angle is being read from "
               "the specified file.")));
        }

        else
        {
          AssertThrow(
            component == dim,
            dealii::ExcMessage(
              ("In 3D there are three euler angles and vectors have "
               "three components. Nonetheless, more or less entries "
               "are being red from the specified files.")));
        }

        write_into.push_back(vector);
      }
    };

  // Read and store the grain orientations
  {
    std::ifstream crystal_orientations_input_file(
      crystal_orientations_file_name + file_extension,
      std::ifstream::in);

    if (crystal_orientations_input_file)
      read_and_store(crystal_orientations_input_file,
                     euler_angles,
                     true);
    else
      AssertThrow(
        false,
        dealii::ExcMessage(
          "File \"" + crystal_orientations_file_name + file_extension +
          "\" not found."));
  }

  AssertThrow(euler_angles.size() == n_crystals,
    dealii::ExcMessage("The triangulation has " +
                       std::to_string(n_crystals) +
                       " crystals but " +
                       std::to_string(euler_angles.size()) +
                       " grain orientations are listed in the "
                       " input file."));

  // Read and store the slip systems from the files
  {
    std::ifstream slip_directions_input_file(
      slip_directions_file_name + file_extension,
      std::ifstream::in);
    std::ifstream slip_normals_input_file(
      slip_normals_file_name + file_extension,
      std::ifstream::in);

    if (slip_directions_input_file)
      read_and_store(slip_directions_input_file,
                     reference_slip_directions,
                     false);
    else
      AssertThrow(
        false,
        dealii::ExcMessage(
          "File \"" + slip_directions_file_name + file_extension +
          "\" not found."));

    if (slip_normals_input_file)
      read_and_store(slip_normals_input_file,
                     reference_slip_normals,
                     false);
    else
      AssertThrow(
        false,
        dealii::ExcMessage(
          "File \"" + slip_normals_file_name + file_extension +
          "\" not found."));
  }

  AssertThrow(reference_slip_normals.size() ==
              reference_slip_directions.size(),
              dealii::ExcMessage("The number of slip normals and "
              "directions in the input files do not match."));

  reference_slip_orthogonals.resize(reference_slip_normals.size(),
                                    dealii::Tensor<1,dim>());

  // Normalize vectors, check for orthogonality and compute the vector
  // completing the orthonormal basis
  for (unsigned int i = 0; i < reference_slip_normals.size(); ++i)
  {
    AssertThrow(orthogonality_check(reference_slip_normals[i],
                                    reference_slip_directions[i]),
                dealii::ExcMessage("The slip normal and the slip "
                                   "direction of the " +
                                   std::to_string(i) + "-th slip "
                                   "system are not orthogonal." ));

    switch (dim)
    {
    case 2:
      reference_slip_orthogonals[i] = 0;
      break;
    case 3:
      reference_slip_orthogonals[i] =
        dealii::cross_product_3d(reference_slip_normals[i],
                                 reference_slip_directions[i]);
      break;
    default:
      AssertThrow(false, dealii::ExcIndexRange(dim,2,3));
      break;
    }

    reference_slip_normals[i]     /= reference_slip_normals[i].norm();
    reference_slip_directions[i]  /= reference_slip_directions[i].norm();

    // Enclosed in switch to avoid NaN values
    switch (dim)
    {
      case 3:
        reference_slip_orthogonals[i] /= reference_slip_orthogonals[i].norm();
        break;
    }
  }

  n_slips = reference_slip_normals.size();
}



template<int dim>
bool CrystalsData<dim>::orthogonality_check(
  const dealii::Tensor<1,dim> a,
  const dealii::Tensor<1,dim> b)
{
  const double scalar_product = dealii::scalar_product(a, b);

  return std::fabs(scalar_product) <= __DBL_EPSILON__;
}



template<int dim>
void CrystalsData<dim>::compute_rotation_matrices()
{
  for (unsigned int crystal_id = 0; crystal_id < n_crystals; crystal_id++)
  {
    dealii::Tensor<2,dim> rotation_tensor;

    const double deg_to_rad = M_PI / 180.0;

    switch (dim)
    {
    case 2:
      {
        const double theta  = euler_angles[crystal_id][0] * deg_to_rad;

        rotation_tensor[0][0] = std::cos(theta);
        rotation_tensor[0][1] = -std::sin(theta);
        rotation_tensor[1][0] = std::sin(theta);
        rotation_tensor[1][1] = std::cos(theta);
      }
      break;
    case 3:
      {
        /*
        const double alpha  = euler_angles[crystal_id][0] * deg_to_rad;
        const double beta   = euler_angles[crystal_id][1] * deg_to_rad;
        const double gamma  = euler_angles[crystal_id][2] * deg_to_rad;
        */

        const double alpha  = euler_angles[crystal_id][0] * deg_to_rad;
        const double beta   = euler_angles[crystal_id][1] * deg_to_rad;
        const double gamma  = (euler_angles[crystal_id][2]) *
                                deg_to_rad;

        dealii::Tensor<2,dim> rotation_tensor_alpha;
        dealii::Tensor<2,dim> rotation_tensor_beta;
        dealii::Tensor<2,dim> rotation_tensor_gamma;

        /*
        rotation_tensor_alpha[0][0] = 1.0;
        rotation_tensor_alpha[1][1] = std::cos(alpha);
        rotation_tensor_alpha[1][2] = -std::sin(alpha);
        rotation_tensor_alpha[2][1] = std::sin(alpha);
        rotation_tensor_alpha[2][2] = std::cos(alpha);

        rotation_tensor_beta[0][0] = std::cos(beta);
        rotation_tensor_beta[0][2] = std::sin(beta);
        rotation_tensor_beta[1][1] = 1.0;
        rotation_tensor_beta[2][0] = -std::sin(beta);
        rotation_tensor_beta[2][2] = std::cos(beta);

        rotation_tensor_gamma[0][0] = std::cos(gamma);
        rotation_tensor_gamma[0][1] = -std::sin(gamma);
        rotation_tensor_gamma[1][0] = std::sin(gamma);
        rotation_tensor_gamma[1][1] = std::cos(gamma);
        rotation_tensor_gamma[2][2] = 1.0;
        */

        rotation_tensor_alpha[0][0] = std::cos(alpha);
        rotation_tensor_alpha[0][1] = -std::sin(alpha);
        rotation_tensor_alpha[1][0] = std::sin(alpha);
        rotation_tensor_alpha[1][1] = std::cos(alpha);
        rotation_tensor_alpha[2][2] = 1.0;

        rotation_tensor_beta[0][0] = 1.0;
        rotation_tensor_beta[1][1] = std::cos(beta);
        rotation_tensor_beta[1][2] = -std::sin(beta);
        rotation_tensor_beta[2][1] = std::sin(beta);
        rotation_tensor_beta[2][2] = std::cos(beta);

        rotation_tensor_gamma[0][0] = std::cos(gamma);
        rotation_tensor_gamma[0][1] = -std::sin(gamma);
        rotation_tensor_gamma[1][0] = std::sin(gamma);
        rotation_tensor_gamma[1][1] = std::cos(gamma);
        rotation_tensor_gamma[2][2] = 1.0;
        /*
        rotation_tensor = dealii::contract<1,0>(
                            rotation_tensor_gamma,
                            dealii::contract<1,0>(
                              rotation_tensor_beta,
                              rotation_tensor_alpha));
        */

        rotation_tensor = rotation_tensor_alpha *
                          rotation_tensor_beta *
                          rotation_tensor_gamma;
      }
      break;
    default:
      AssertThrow(false, dealii::ExcIndexRange(dim,2,3));
      break;
    }

    rotation_tensors.push_back(rotation_tensor);
  }
}



template<int dim>
void CrystalsData<dim>::compute_3d_rotation_matrices()
{
  if constexpr(dim == 3)
    rotation_tensors_3d = rotation_tensors;
  else if constexpr (dim == 2)
    for (unsigned int crystal_id = 0;
        crystal_id < n_crystals; crystal_id++)
    {
      dealii::Tensor<2,3> rotation_tensor;

      const double deg_to_rad = M_PI / 180.0;

      const double alpha  = 0.0;
      const double beta   = 0.0;
      const double gamma  = euler_angles[crystal_id][0] * deg_to_rad;

      dealii::Tensor<2,3> rotation_tensor_alpha;
      dealii::Tensor<2,3> rotation_tensor_beta;
      dealii::Tensor<2,3> rotation_tensor_gamma;

      rotation_tensor_alpha[0][0] = 1.0;
      rotation_tensor_alpha[1][1] = std::cos(alpha);
      rotation_tensor_alpha[1][2] = -std::sin(alpha);
      rotation_tensor_alpha[2][1] = std::sin(alpha);
      rotation_tensor_alpha[2][2] = std::cos(alpha);

      rotation_tensor_beta[0][0] = std::cos(beta);
      rotation_tensor_beta[0][2] = std::sin(beta);
      rotation_tensor_beta[1][1] = 1.0;
      rotation_tensor_beta[2][0] = -std::sin(beta);
      rotation_tensor_beta[2][2] = std::cos(beta);

      rotation_tensor_gamma[0][0] = std::cos(gamma);
      rotation_tensor_gamma[0][1] = -std::sin(gamma);
      rotation_tensor_gamma[1][0] = std::sin(gamma);
      rotation_tensor_gamma[1][1] = std::cos(gamma);
      rotation_tensor_gamma[2][2] = 1.0;

      rotation_tensor = dealii::contract<1,0>(
                          rotation_tensor_gamma,
                          dealii::contract<1,0>(
                            rotation_tensor_beta,
                            rotation_tensor_alpha));

      rotation_tensors_3d.push_back(rotation_tensor);
    }
  else
    Assert(false, dealii::ExcNotImplemented());
}



template<int dim>
void CrystalsData<dim>::compute_slip_systems()
{
  for (unsigned int crystal_id = 0; crystal_id < n_crystals; crystal_id++)
  {
    std::vector<dealii::Tensor<1,dim>>  rotated_slip_directions(n_slips);
    std::vector<dealii::Tensor<1,dim>>  rotated_slip_normals(n_slips);
    std::vector<dealii::Tensor<1,dim>>  rotated_slip_orthogonals(n_slips);
    std::vector<dealii::Tensor<2,dim>>  rotated_schmid_tensor(n_slips);
    std::vector<dealii::SymmetricTensor<2,dim>>
                                        rotated_symmetrized_schmid_tensor(n_slips);

    for (unsigned int slip_id = 0; slip_id < n_slips; slip_id++)
    {
      rotated_slip_directions[slip_id] =
        rotation_tensors[crystal_id] *
        reference_slip_directions[slip_id];
      rotated_slip_normals[slip_id] =
        rotation_tensors[crystal_id] *
        reference_slip_normals[slip_id];
      rotated_slip_orthogonals[slip_id] =
        rotation_tensors[crystal_id] *
        reference_slip_orthogonals[slip_id];

      rotated_schmid_tensor[slip_id] =
        dealii::outer_product(rotated_slip_directions[slip_id],
                              rotated_slip_normals[slip_id]);

      rotated_symmetrized_schmid_tensor[slip_id] =
        dealii::symmetrize(rotated_schmid_tensor[slip_id]);
    }

    slip_directions.push_back(rotated_slip_directions);
    slip_normals.push_back(rotated_slip_normals);
    slip_orthogonals.push_back(rotated_slip_orthogonals);
    schmid_tensors.push_back(rotated_schmid_tensor);
    symmetrized_schmid_tensors.push_back(rotated_symmetrized_schmid_tensor);
  }
}



} // gCP



template class gCP::CrystalsData<2>;
template class gCP::CrystalsData<3>;


