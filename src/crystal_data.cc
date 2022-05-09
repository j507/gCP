#include <gCP/crystal_data.h>
#include <deal.II/base/tensor.h>
#include <fstream>

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
        std::vector<dealii::Tensor<1,dim>>  &write_into)
    {
      unsigned int j;

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

        j = 0;

        while(std::getline(line_as_stream_input,
                            vector_component,
                            ','))
        {
          //dealii::AssertIndexRange(j, dim);

          vector[j++] = std::stod(vector_component);
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
                    euler_angles);
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
                    reference_slip_directions);
    else
      AssertThrow(
        false,
        dealii::ExcMessage(
          "File \"" + slip_directions_file_name + file_extension +
          "\" not found."));

    if (slip_normals_input_file)
      read_and_store(slip_normals_input_file,
                    reference_slip_normals);
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
      break;
    }

    reference_slip_normals[i]     /= reference_slip_normals[i].norm();
    reference_slip_directions[i]  /= reference_slip_directions[i].norm();
    reference_slip_orthogonals[i] /= reference_slip_orthogonals[i].norm();
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
    dealii::Tensor<2,dim> rotation_matrix;

    switch (dim)
    {
    case 2:
      {
        const double theta  = euler_angles[crystal_id][0];

        rotation_matrix[0][0] = std::cos(theta);
        rotation_matrix[0][1] = -std::sin(theta);
        rotation_matrix[1][0] = std::sin(theta);
        rotation_matrix[1][1] = std::cos(theta);
      }
      break;
    case 3:
      {
        const double alpha  = euler_angles[crystal_id][0];
        const double beta   = euler_angles[crystal_id][1];
        const double gamma  = euler_angles[crystal_id][2];

        dealii::Tensor<2,dim> rotation_matrix_alpha;
        dealii::Tensor<2,dim> rotation_matrix_beta;
        dealii::Tensor<2,dim> rotation_matrix_gamma;

        rotation_matrix_alpha[0][0] = 1.0;
        rotation_matrix_alpha[1][1] = std::cos(alpha);
        rotation_matrix_alpha[1][2] = -std::sin(alpha);
        rotation_matrix_alpha[2][1] = std::sin(alpha);
        rotation_matrix_alpha[2][2] = std::cos(alpha);

        rotation_matrix_beta[0][0] = std::cos(beta);
        rotation_matrix_beta[0][2] = std::sin(beta);
        rotation_matrix_beta[1][1] = 1.0;
        rotation_matrix_beta[2][0] = -std::sin(beta);
        rotation_matrix_beta[2][2] = std::cos(beta);

        rotation_matrix_gamma[0][0] = std::cos(gamma);
        rotation_matrix_gamma[0][1] = -std::sin(gamma);
        rotation_matrix_gamma[1][0] = std::sin(gamma);
        rotation_matrix_gamma[1][1] = std::cos(gamma);
        rotation_matrix_gamma[2][2] = 1.0;

        rotation_matrix = dealii::contract<1,0>(
                            rotation_matrix_gamma,
                            dealii::contract<1,0>(
                              rotation_matrix_beta,
                              rotation_matrix_alpha));
      }
      break;
    default:
      dealii::ExcInternalError();
      break;
    }

    rotation_matrices.push_back(rotation_matrix);
  }
}



template<int dim>
void CrystalsData<dim>::compute_slip_systems()
{
  for (unsigned int crystal_id = 0; crystal_id < n_crystals; crystal_id++)
  {
    std::vector<dealii::Tensor<1,dim>> rotated_slip_directions(n_slips);
    std::vector<dealii::Tensor<1,dim>> rotated_slip_normals(n_slips);
    std::vector<dealii::Tensor<1,dim>> rotated_slip_orthogonals(n_slips);
    std::vector<dealii::Tensor<2,dim>> rotated_schmid_tensor(n_slips);
    std::vector<dealii::Tensor<2,dim>> rotated_symmetrized_schmid_tensor(n_slips);


    for (unsigned int slip_id = 0; slip_id < n_slips; slip_id++)
    {
      rotated_slip_directions[slip_id] =
        rotation_matrices[crystal_id] *
        reference_slip_directions[slip_id];
      rotated_slip_normals[slip_id] =
        rotation_matrices[crystal_id] *
        reference_slip_normals[slip_id];
      rotated_slip_orthogonals[slip_id] =
        rotation_matrices[crystal_id] *
        reference_slip_orthogonals[slip_id];

      rotated_schmid_tensor[slip_id] =
        dealii::outer_product(rotated_slip_directions[slip_id],
                              rotated_slip_normals[slip_id]);

      rotated_symmetrized_schmid_tensor[slip_id] =
        dealii::symmetrize(rotated_schmid_tensor[slip_id]);
    }

    slip_directions.push_back(rotated_slip_normals);
    slip_normals.push_back(rotated_slip_normals);
    slip_orthogonals.push_back(rotated_slip_orthogonals);
    schmid_tensors.push_back(rotated_schmid_tensor);
    symmetrized_schmid_tensors.push_back(rotated_symmetrized_schmid_tensor);
  }
}



} // gCP



template class gCP::CrystalsData<2>;
template class gCP::CrystalsData<3>;


