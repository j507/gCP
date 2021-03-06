#include <gCP/constitutive_laws.h>

#include <deal.II/base/symmetric_tensor.h>

#include <iomanip>

namespace gCP
{



namespace Kinematics
{



template <int dim>
ElasticStrain<dim>::ElasticStrain(
  std::shared_ptr<CrystalsData<dim>>  crystals_data)
:
crystals_data(crystals_data)
{}



template <int dim>
const dealii::SymmetricTensor<2,dim>
ElasticStrain<dim>::get_elastic_strain_tensor(
  const unsigned int                      crystal_id,
  const unsigned int                      q_point,
  const dealii::SymmetricTensor<2,dim>    strain_tensor_value,
  const std::vector<std::vector<double>>  slip_values) const
{
  AssertThrow(crystals_data->is_initialized(),
              dealii::ExcMessage("The underlying CrystalsData<dim>"
                                  " instance has not been "
                                  " initialized."));

  dealii::SymmetricTensor<2,dim> elastic_strain_tensor_value(
                                  strain_tensor_value);

  for (unsigned int slip_id = 0;
       slip_id < crystals_data->get_n_slips();
       ++slip_id)
  {
    elastic_strain_tensor_value -=
      slip_values[slip_id][q_point] *
      crystals_data->get_symmetrized_schmid_tensor(crystal_id, slip_id);
  }

  return elastic_strain_tensor_value;
}



} // namespace Kinematics



namespace ConstitutiveLaws
{



template<int dim>
HookeLaw<dim>::HookeLaw(
  const RunTimeParameters::HookeLawParameters  parameters)
:
crystallite(Crystallite::Monocrystalline),
C1111(parameters.C1111),
C1122(parameters.C1122),
C1212(parameters.C1212),
flag_init_was_called(false)
{
  crystals_data = nullptr;
}



template<int dim>
HookeLaw<dim>::HookeLaw(
  const std::shared_ptr<CrystalsData<dim>>    &crystals_data,
  const RunTimeParameters::HookeLawParameters parameters)
:
crystals_data(crystals_data),
crystallite(Crystallite::Polycrystalline),
C1111(parameters.C1111),
C1122(parameters.C1122),
C1212(parameters.C1212),
flag_init_was_called(false)
{}



template<int dim>
void HookeLaw<dim>::init()
{
  for (unsigned int i = 0; i < dim; i++)
    for (unsigned int j = 0; j < dim; j++)
      for (unsigned int k = 0; k < dim; k++)
        for (unsigned int l = 0; l < dim; l++)
          if (i == j && j == k && k == l)
            reference_stiffness_tetrad[i][j][k][l] = C1111;
          else if (i == k && j == l)
            reference_stiffness_tetrad[i][j][k][l] = C1212;
          else if (i == j && k == l)
            reference_stiffness_tetrad[i][j][k][l] = C1122;

  switch (crystallite)
  {
  case Crystallite::Monocrystalline:
    break;

  case Crystallite::Polycrystalline:
    {
      AssertThrow(crystals_data->is_initialized(),
                  dealii::ExcMessage("The underlying CrystalsData<dim>"
                                     " instance has not been "
                                     " initialized."));

      for (unsigned int crystal_id = 0;
           crystal_id < crystals_data->get_n_crystals();
           crystal_id++)
        {
          dealii::SymmetricTensor<4,dim> stiffness_tetrad;

          dealii::Tensor<2,dim> rotation_tensor =
            crystals_data->get_rotation_tensor(crystal_id);

          // The indices j and l do not start at zero due to the
          // nature of the dealii::SymmetricTensor<4,dim> class, where
          // for example [0][1][2][1] points to the same memory location
          // as [1][0][2][1]
          for (unsigned int i = 0; i < dim; i++)
            for (unsigned int j = i; j < dim; j++)
              for (unsigned int k = 0; k < dim; k++)
                for (unsigned int l = k; l < dim; l++)
                  for (unsigned int o = 0; o < dim; o++)
                    for (unsigned int p = 0; p < dim; p++)
                      for (unsigned int q = 0; q < dim; q++)
                        for (unsigned int r = 0; r < dim; r++)
                          stiffness_tetrad[i][j][k][l] +=
                            rotation_tensor[i][o] *
                            rotation_tensor[j][p] *
                            rotation_tensor[k][q] *
                            rotation_tensor[l][r] *
                            reference_stiffness_tetrad[o][p][q][r];

          stiffness_tetrads.push_back(stiffness_tetrad);
        }
    }
    break;

  default:
    AssertThrow(false, dealii::ExcNotImplemented())
    break;
  }

  flag_init_was_called = true;
}


template<int dim>
const dealii::SymmetricTensor<2,dim> HookeLaw<dim>::
get_stress_tensor(
  const dealii::SymmetricTensor<2,dim> strain_tensor_values) const
{
  AssertThrow(crystallite == Crystallite::Monocrystalline,
              dealii::ExcMessage("This overload is meant for the"
                                 " case of a monocrystalline."
                                 " Nonetheless a CrystalsData<dim>'s"
                                 " shared pointer was passed on to the"
                                 " constructor"));

  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The HookeLaw<dim> instance has not"
                                 " been initialized."));

  return reference_stiffness_tetrad * strain_tensor_values;
}



template<int dim>
const dealii::SymmetricTensor<2,dim> HookeLaw<dim>::
get_stress_tensor(
  const unsigned int                    crystal_id,
  const dealii::SymmetricTensor<2,dim>  strain_tensor_values) const
{
  AssertThrow(crystallite == Crystallite::Polycrystalline,
              dealii::ExcMessage("This overload is meant for the"
                                 " case of a polycrystalline."
                                 " Nonetheless no CrystalsData<dim>'s"
                                 " shared pointer was passed on to the"
                                 " constructor"));

  AssertThrow(crystals_data.get() != nullptr,
              dealii::ExcMessage("This overloaded method requires a "
                                 "constructor call where a "
                                 "CrystalsData<dim> instance is "
                                 "passed as a std::shared_ptr"))

  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The HookeLaw<dim> instance has not"
                                 " been initialized."));

  AssertIndexRange(crystal_id, crystals_data->get_n_crystals());

  return stiffness_tetrads[crystal_id] * strain_tensor_values;
}



template<int dim>
ResolvedShearStressLaw<dim>::ResolvedShearStressLaw(
  const std::shared_ptr<CrystalsData<dim>> &crystals_data)
:
crystals_data(crystals_data)
{}



template<int dim>
ScalarMicroscopicStressLaw<dim>::ScalarMicroscopicStressLaw(
  const std::shared_ptr<CrystalsData<dim>>                      &crystals_data,
  const RunTimeParameters::ScalarMicroscopicStressLawParameters parameters)
:
crystals_data(crystals_data),
regularization_function(parameters.regularization_function),
regularization_parameter(parameters.regularization_parameter),
initial_slip_resistance(parameters.initial_slip_resistance),
linear_hardening_modulus(parameters.linear_hardening_modulus),
hardening_parameter(parameters.hardening_parameter)
{}



template<int dim>
double ScalarMicroscopicStressLaw<dim>::get_scalar_microscopic_stress(
  const double slip_value,
  const double old_slip_value,
  const double slip_resistance,
  const double time_step_size)
{
  AssertThrow(crystals_data->is_initialized(),
              dealii::ExcMessage("The underlying CrystalsData<dim>"
                                  " instance has not been "
                                  " initialized."));

  double regularization_factor;

  const double slip_rate = (slip_value - old_slip_value) /
                            time_step_size;

  switch (regularization_function)
  {
  case RunTimeParameters::RegularizationFunction::PowerLaw:
    {
      regularization_factor = std::pow(slip_rate,
                                       1.0 / regularization_parameter);
    }
    break;
  case RunTimeParameters::RegularizationFunction::Tanh:
    {
      regularization_factor = std::tanh(slip_rate /
                                        regularization_parameter);
    }
    break;
  default:
    {
      AssertThrow(false, dealii::ExcMessage("The given regularization "
                                            "function is not currently "
                                            "implemented."));
    }
    break;
  }

  AssertIsFinite(slip_resistance);
  AssertIsFinite(regularization_factor);

  return ((initial_slip_resistance + slip_resistance) *
          regularization_factor);
}



template<int dim>
dealii::FullMatrix<double> ScalarMicroscopicStressLaw<dim>::
  get_gateaux_derivative_matrix(
    const unsigned int                      q_point,
    const std::vector<std::vector<double>>  slip_values,
    const std::vector<std::vector<double>>  old_slip_values,
    const std::vector<double>               slip_resistances,
    const double                            time_step_size)
{
  AssertThrow(crystals_data->is_initialized(),
              dealii::ExcMessage("The underlying CrystalsData<dim>"
                                  " instance has not been "
                                  " initialized."));

  dealii::FullMatrix<double> matrix(crystals_data->get_n_slips());

  auto compute_slip_rate =
    [&slip_values, &old_slip_values, &time_step_size](
      const unsigned int  q_point,
      const unsigned int  slip_id)
  {
    const double slip_rate = (slip_values[slip_id][q_point] -
            old_slip_values[slip_id][q_point]) / time_step_size;

    AssertIsFinite(slip_rate);
    return slip_rate;
  };

  for (unsigned int slip_id_alpha = 0;
      slip_id_alpha < crystals_data->get_n_slips();
      ++slip_id_alpha)
    for (unsigned int slip_id_beta = 0;
        slip_id_beta < crystals_data->get_n_slips();
        ++slip_id_beta)
    {
      matrix[slip_id_alpha][slip_id_beta] =
        (get_hardening_matrix_entry(slip_id_alpha == slip_id_beta) *
          get_regularization_factor(
            compute_slip_rate(q_point, slip_id_alpha)) *
          sgn(compute_slip_rate(q_point, slip_id_beta)));

      if (slip_id_alpha == slip_id_beta)
        matrix[slip_id_alpha][slip_id_beta] +=
          ((initial_slip_resistance +
            slip_resistances[slip_id_alpha]) /
            (time_step_size * regularization_parameter) *
            std::pow(1.0/std::cosh(
                      compute_slip_rate(q_point, slip_id_alpha) /
                      regularization_parameter), 2));

      AssertIsFinite(matrix[slip_id_alpha][slip_id_beta]);
    }

  return matrix;
}



template <int dim>
double ScalarMicroscopicStressLaw<dim>::
get_regularization_factor(const double slip_rate) const
{
  double regularization_factor = 0.0;

  switch (regularization_function)
  {
  case RunTimeParameters::RegularizationFunction::PowerLaw:
    {
      regularization_factor = std::pow(slip_rate,
                                       1.0 / regularization_parameter);
    }
    break;
  case RunTimeParameters::RegularizationFunction::Tanh:
    {
      regularization_factor = std::tanh(slip_rate /
                                        regularization_parameter);
    }
    break;
  default:
    {
      AssertThrow(false, dealii::ExcMessage("The given regularization "
                                            "function is not currently "
                                            "implemented."));
    }
    break;
  }

  AssertIsFinite(regularization_factor);

  return regularization_factor;
}



template<int dim>
VectorMicroscopicStressLaw<dim>::VectorMicroscopicStressLaw(
  const std::shared_ptr<CrystalsData<dim>> &crystals_data,
  const RunTimeParameters::VectorMicroscopicStressLawParameters parameters)
:
crystals_data(crystals_data),
energetic_length_scale(parameters.energetic_length_scale),
initial_slip_resistance(parameters.initial_slip_resistance),
flag_init_was_called(false)
{}



template<int dim>
void VectorMicroscopicStressLaw<dim>::init()
{
  AssertThrow(crystals_data->is_initialized(),
              dealii::ExcMessage("The underlying CrystalsData<dim>"
                                  " instance has not been "
                                  " initialized."));

  for (unsigned int crystal_id = 0;
        crystal_id < crystals_data->get_n_crystals();
        crystal_id++)
  {
    std::vector<dealii::SymmetricTensor<2,dim>>
      reduced_gradient_hardening_tensors_per_crystal;

    for (unsigned int slip_id = 0;
          slip_id < crystals_data->get_n_slips();
          ++slip_id)
    {
      const dealii::Tensor<1,dim> slip_direction =
        crystals_data->get_slip_direction(crystal_id, slip_id);

      const dealii::Tensor<1,dim> slip_orthogonal =
        crystals_data->get_slip_orthogonal(crystal_id, slip_id);

      const dealii::SymmetricTensor<2,dim> slip_direction_outer_product =
        dealii::symmetrize(dealii::outer_product(slip_direction,
                                                 slip_direction));

      const dealii::SymmetricTensor<2,dim> slip_orthogonal_outer_product =
        dealii::symmetrize(dealii::outer_product(slip_orthogonal,
                                                 slip_orthogonal));

      const dealii::SymmetricTensor<2,dim> reduced_gradient_hardening_tensor =
        initial_slip_resistance *
        energetic_length_scale * energetic_length_scale *
        (slip_direction_outer_product + slip_orthogonal_outer_product);

      for (unsigned int i = 0;
           i < reduced_gradient_hardening_tensor.n_independent_components;
           ++i)
        AssertIsFinite(reduced_gradient_hardening_tensor.access_raw_entry(i));

      reduced_gradient_hardening_tensors_per_crystal.push_back(
        reduced_gradient_hardening_tensor);
    }

    reduced_gradient_hardening_tensors.push_back(
      reduced_gradient_hardening_tensors_per_crystal);
  }

  flag_init_was_called = true;
}



template <int dim>
dealii::Tensor<1,dim> VectorMicroscopicStressLaw<dim>::
get_vector_microscopic_stress(
  const unsigned int          crystal_id,
  const unsigned int          slip_id,
  const dealii::Tensor<1,dim> slip_gradient) const
{
  AssertIndexRange(crystal_id, crystals_data->get_n_crystals());
  AssertIndexRange(slip_id, crystals_data->get_n_slips());

  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The VectorMicroscopicStressLaw<dim> "
                                 "instance has not been initialized."));

  return (reduced_gradient_hardening_tensors[crystal_id][slip_id] *
          slip_gradient);
}



template<int dim>
MicroscopicTractionLaw<dim>::MicroscopicTractionLaw(
  const std::shared_ptr<CrystalsData<dim>> &crystals_data,
  const RunTimeParameters::MicroscopicTractionLawParameters parameters)
:
crystals_data(crystals_data),
grain_boundary_modulus(parameters.grain_boundary_modulus)
{}



template<>
MicroscopicTractionLaw<2>::GrainInteractionModuli
MicroscopicTractionLaw<2>::get_grain_interaction_moduli(
  const unsigned int                crystal_id_current_cell,
  const unsigned int                crystal_id_neighbour_cell,
  std::vector<dealii::Tensor<1,2>>  normal_vector_values) const
{
  AssertThrow(crystal_id_current_cell != crystal_id_neighbour_cell,
              dealii::ExcMessage(
                "The crystal identifiers match. This method only "
                "meant to be used at grain boundaries"));

  const unsigned int n_q_points = normal_vector_values.size();

  std::vector<dealii::FullMatrix<double>> intra_grain_interaction_moduli;

  std::vector<dealii::FullMatrix<double>> inter_grain_interaction_moduli;

  intra_grain_interaction_moduli.reserve(n_q_points);
  inter_grain_interaction_moduli.reserve(n_q_points);

  dealii::FullMatrix<double> intra_grain_interaction_moduli_per_q_point(
    crystals_data->get_n_slips());
  dealii::FullMatrix<double> inter_grain_interaction_moduli_per_q_point(
    crystals_data->get_n_slips());

  // Get slip systems of the current cell
  std::vector<dealii::Tensor<1,2>> slip_directions_current_cell =
    crystals_data->get_slip_directions(crystal_id_current_cell);
  std::vector<dealii::Tensor<1,2>> slip_normals_current_cell =
    crystals_data->get_slip_normals(crystal_id_current_cell);

  // Get slip systems of the neighbour cell
  std::vector<dealii::Tensor<1,2>> slip_directions_neighbour_cell =
    crystals_data->get_slip_directions(crystal_id_neighbour_cell);
  std::vector<dealii::Tensor<1,2>> slip_normals_neighbour_cell =
    crystals_data->get_slip_normals(crystal_id_neighbour_cell);

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
  {
    intra_grain_interaction_moduli_per_q_point = 0.0;
    inter_grain_interaction_moduli_per_q_point = 0.0;

    for (unsigned int slip_id_alpha = 0;
         slip_id_alpha < crystals_data->get_n_slips(); ++slip_id_alpha)
      for (unsigned int slip_id_beta = 0;
           slip_id_beta < crystals_data->get_n_slips(); ++slip_id_beta)
      {
        intra_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta] =
          (slip_directions_current_cell[slip_id_alpha] *
           slip_directions_current_cell[slip_id_beta]) *
          (slip_normals_current_cell[slip_id_alpha][0] *
           normal_vector_values[q_point][1] -
           slip_normals_current_cell[slip_id_alpha][1] *
           normal_vector_values[q_point][0]) *
          (slip_normals_current_cell[slip_id_beta][0] *
           normal_vector_values[q_point][1] -
           slip_normals_current_cell[slip_id_beta][1] *
           normal_vector_values[q_point][0]);

        inter_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta] =
          (slip_directions_current_cell[slip_id_alpha] *
           slip_directions_neighbour_cell[slip_id_beta]) *
          (slip_normals_current_cell[slip_id_alpha][0] *
           normal_vector_values[q_point][1] -
           slip_normals_current_cell[slip_id_alpha][1] *
           normal_vector_values[q_point][0]) *
          (slip_normals_neighbour_cell[slip_id_beta][0] *
           normal_vector_values[q_point][1] -
           slip_normals_neighbour_cell[slip_id_beta][1] *
           normal_vector_values[q_point][0]);

        AssertThrow(
          std::fabs(intra_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]) >= 0.0 &&
            std::fabs(intra_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]) <= 1.0,
          dealii::ExcMessage(
            "The interaction moduli should be inside the "
            "range [0,1]."));

        AssertThrow(
          std::fabs(inter_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]) >= 0.0 &&
            std::fabs(inter_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]) <= 1.0,
          dealii::ExcMessage(
            "The interaction moduli should be inside the "
            "range [0,1]."));

        AssertIsFinite(
          intra_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]);
        AssertIsFinite(
          inter_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]);
      }

    intra_grain_interaction_moduli.push_back(intra_grain_interaction_moduli_per_q_point);
    inter_grain_interaction_moduli.push_back(inter_grain_interaction_moduli_per_q_point);
  }

  AssertThrow(normal_vector_values.size() ==
              intra_grain_interaction_moduli.size(),
              dealii::ExcDimensionMismatch(
                normal_vector_values.size(),
                intra_grain_interaction_moduli.size()));
  AssertThrow(normal_vector_values.size() ==
              inter_grain_interaction_moduli.size(),
              dealii::ExcDimensionMismatch(
                normal_vector_values.size(),
                inter_grain_interaction_moduli.size()));

  return (std::make_pair(intra_grain_interaction_moduli,
                         inter_grain_interaction_moduli));
}



template<>
MicroscopicTractionLaw<3>::GrainInteractionModuli
MicroscopicTractionLaw<3>::get_grain_interaction_moduli(
  const unsigned int                crystal_id_current_cell,
  const unsigned int                crystal_id_neighbour_cell,
  std::vector<dealii::Tensor<1,3>>  normal_vector_values) const
{
  AssertThrow(crystal_id_current_cell != crystal_id_neighbour_cell,
              dealii::ExcMessage(
                "The crystal identifiers match. This method only "
                "meant to be used at grain boundaries"));

  const unsigned int n_q_points = normal_vector_values.size();

  std::vector<dealii::FullMatrix<double>> intra_grain_interaction_moduli;

  std::vector<dealii::FullMatrix<double>> inter_grain_interaction_moduli;

  intra_grain_interaction_moduli.reserve(n_q_points);
  inter_grain_interaction_moduli.reserve(n_q_points);

  dealii::FullMatrix<double> intra_grain_interaction_moduli_per_q_point(
    crystals_data->get_n_slips());
  dealii::FullMatrix<double> inter_grain_interaction_moduli_per_q_point(
    crystals_data->get_n_slips());

  // Get slip systems of the current cell
  std::vector<dealii::Tensor<1,3>> slip_directions_current_cell =
    crystals_data->get_slip_directions(crystal_id_current_cell);
  std::vector<dealii::Tensor<1,3>> slip_normals_current_cell =
    crystals_data->get_slip_normals(crystal_id_current_cell);

  // Get slip systems of the neighbour cell
  std::vector<dealii::Tensor<1,3>> slip_directions_neighbour_cell =
    crystals_data->get_slip_directions(crystal_id_neighbour_cell);
  std::vector<dealii::Tensor<1,3>> slip_normals_neighbour_cell =
    crystals_data->get_slip_normals(crystal_id_neighbour_cell);

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
  {
    intra_grain_interaction_moduli_per_q_point = 0.0;
    inter_grain_interaction_moduli_per_q_point = 0.0;

    for (unsigned int slip_id_alpha = 0;
         slip_id_alpha < crystals_data->get_n_slips(); ++slip_id_alpha)
      for (unsigned int slip_id_beta = 0;
           slip_id_beta < crystals_data->get_n_slips(); ++slip_id_beta)
      {
        intra_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta] =
          (slip_directions_current_cell[slip_id_alpha] *
           slip_directions_current_cell[slip_id_beta]) *
          (dealii::cross_product_3d(
            slip_normals_current_cell[slip_id_alpha],
            normal_vector_values[q_point]) *
           dealii::cross_product_3d(
            slip_normals_current_cell[slip_id_beta],
            normal_vector_values[q_point]));

        inter_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta] =
          (slip_directions_current_cell[slip_id_alpha] *
           slip_directions_neighbour_cell[slip_id_beta]) *
          (dealii::cross_product_3d(
            slip_normals_current_cell[slip_id_alpha],
            normal_vector_values[q_point]) *
           dealii::cross_product_3d(
            slip_normals_neighbour_cell[slip_id_beta],
            normal_vector_values[q_point]));

        AssertThrow(
          std::fabs(intra_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]) >= 0.0 &&
            std::fabs(intra_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]) <= 1.0,
          dealii::ExcMessage(
            "The interaction moduli should be inside the "
            "range [0,1]."));

        AssertThrow(
          std::fabs(inter_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]) >= 0.0 &&
            std::fabs(inter_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]) <= 1.0,
          dealii::ExcMessage(
            "The interaction moduli should be inside the "
            "range [0,1]."));

        AssertIsFinite(
          intra_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]);
        AssertIsFinite(
          inter_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]);
      }

    intra_grain_interaction_moduli.push_back(intra_grain_interaction_moduli_per_q_point);
    inter_grain_interaction_moduli.push_back(inter_grain_interaction_moduli_per_q_point);
  }

  AssertThrow(normal_vector_values.size() ==
              intra_grain_interaction_moduli.size(),
              dealii::ExcDimensionMismatch(
                normal_vector_values.size(),
                intra_grain_interaction_moduli.size()));
  AssertThrow(normal_vector_values.size() ==
              inter_grain_interaction_moduli.size(),
              dealii::ExcDimensionMismatch(
                normal_vector_values.size(),
                inter_grain_interaction_moduli.size()));

  return (std::make_pair(intra_grain_interaction_moduli,
                         inter_grain_interaction_moduli));
}



template <int dim>
double MicroscopicTractionLaw<dim>::get_microscopic_traction(
  const unsigned int                      q_point,
  const unsigned int                      slip_id_alpha,
  const GrainInteractionModuli            grain_interaction_moduli,
  const std::vector<std::vector<double>>  slip_values_current_cell,
  const std::vector<std::vector<double>>  slip_values_neighbour_cell) const
{
  double microscopic_traction = 0.0;

  for (unsigned int slip_id_beta = 0;
       slip_id_beta < crystals_data->get_n_slips(); slip_id_beta++)
    microscopic_traction +=
      grain_interaction_moduli.first[q_point][slip_id_alpha][slip_id_beta] *
      slip_values_current_cell[slip_id_beta][q_point]
      -
      grain_interaction_moduli.second[q_point][slip_id_alpha][slip_id_beta] *
      slip_values_neighbour_cell[slip_id_beta][q_point];

  microscopic_traction *= -grain_boundary_modulus;

  return (microscopic_traction);
}



template <int dim>
const dealii::FullMatrix<double>
MicroscopicTractionLaw<dim>::
  get_intra_gateaux_derivative(
    const unsigned int            q_point,
    const GrainInteractionModuli  grain_interaction_moduli) const
{
  dealii::FullMatrix<double> intra_gateaux_derivative =
    grain_interaction_moduli.first[q_point];

  AssertThrow(
    intra_gateaux_derivative.m() ==
      intra_gateaux_derivative.n(),
    dealii::ExcMessage(
      "The dealii::FullMatrix<double> is not square"));

  intra_gateaux_derivative *= -grain_boundary_modulus;

  return (intra_gateaux_derivative);
}



template <int dim>
const dealii::FullMatrix<double>
MicroscopicTractionLaw<dim>::
  get_inter_gateaux_derivative(
    const unsigned int            q_point,
    const GrainInteractionModuli  grain_interaction_moduli) const
{
  dealii::FullMatrix<double> inter_gateaux_derivative =
    grain_interaction_moduli.second[q_point];

  AssertThrow(
    inter_gateaux_derivative.m() ==
      inter_gateaux_derivative.n(),
    dealii::ExcMessage(
      "The dealii::FullMatrix<double> is not square"));

  inter_gateaux_derivative *= grain_boundary_modulus;

  return (inter_gateaux_derivative);
}



} // ConstitutiveLaws



} // gCP


template class gCP::Kinematics::ElasticStrain<2>;
template class gCP::Kinematics::ElasticStrain<3>;

template class gCP::ConstitutiveLaws::HookeLaw<2>;
template class gCP::ConstitutiveLaws::HookeLaw<3>;

template class gCP::ConstitutiveLaws::ResolvedShearStressLaw<2>;
template class gCP::ConstitutiveLaws::ResolvedShearStressLaw<3>;

template class gCP::ConstitutiveLaws::ScalarMicroscopicStressLaw<2>;
template class gCP::ConstitutiveLaws::ScalarMicroscopicStressLaw<3>;

template class gCP::ConstitutiveLaws::VectorMicroscopicStressLaw<2>;
template class gCP::ConstitutiveLaws::VectorMicroscopicStressLaw<3>;

template class gCP::ConstitutiveLaws::MicroscopicTractionLaw<2>;
template class gCP::ConstitutiveLaws::MicroscopicTractionLaw<3>;