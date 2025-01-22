#include <gCP/constitutive_laws.h>

#include <deal.II/base/symmetric_tensor.h>

#include <iomanip>

namespace gCP
{



namespace Kinematics
{



template <int dim>
ElasticStrain<dim>::ElasticStrain(
  std::shared_ptr<CrystalsData<dim>>  crystals_data,
  const double dimensionless_number)
:
crystals_data(crystals_data),
dimensionless_number(dimensionless_number)
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

  for (unsigned int slip_id = 0; slip_id < crystals_data->get_n_slips();
        ++slip_id)
  {
    elastic_strain_tensor_value -=
      dimensionless_number *
      slip_values[slip_id][q_point] *
      crystals_data->get_symmetrized_schmid_tensor(crystal_id, slip_id);
  }

  return elastic_strain_tensor_value;
}



template <int dim>
const dealii::SymmetricTensor<2,dim>
ElasticStrain<dim>::get_plastic_strain_tensor(
  const unsigned int                      crystal_id,
  const unsigned int                      q_point,
  const std::vector<std::vector<double>>  slip_values) const
{
  AssertThrow(crystals_data->is_initialized(),
              dealii::ExcMessage("The underlying CrystalsData<dim>"
                                  " instance has not been "
                                  " initialized."));

  dealii::SymmetricTensor<2,dim> plastic_strain_tensor;

  for (unsigned int slip_id = 0;
       slip_id < crystals_data->get_n_slips();
       ++slip_id)
  {
    plastic_strain_tensor +=
      slip_values[slip_id][q_point] *
      crystals_data->get_symmetrized_schmid_tensor(crystal_id, slip_id);
  }

  return plastic_strain_tensor;
}



} // namespace Kinematics



namespace ConstitutiveLaws
{



template<int dim>
HookeLaw<dim>::HookeLaw(
  const RunTimeParameters::HookeLawParameters  parameters,
  const double characteristic_stiffness)
:
crystallite(Crystallite::Monocrystalline),
C1111(parameters.C1111),
C1122(parameters.C1122),
C1212(parameters.C1212),
characteristic_stiffness(characteristic_stiffness),
flag_init_was_called(false)
{
  crystals_data = nullptr;
}



template<int dim>
HookeLaw<dim>::HookeLaw(
  const std::shared_ptr<CrystalsData<dim>>    &crystals_data,
  const RunTimeParameters::HookeLawParameters parameters,
  const double characteristic_stiffness)
:
crystals_data(crystals_data),
crystallite(Crystallite::Polycrystalline),
C1111(parameters.C1111),
C1122(parameters.C1122),
C1212(parameters.C1212),
characteristic_stiffness(characteristic_stiffness),
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
            reference_stiffness_tetrad[i][j][k][l] =
              C1111 / characteristic_stiffness;
          else if (i == k && j == l)
            reference_stiffness_tetrad[i][j][k][l] =
              C1212 / characteristic_stiffness;
          else if (i == j && k == l)
            reference_stiffness_tetrad[i][j][k][l] =
              C1122 / characteristic_stiffness;

  if constexpr(dim == 3)
    reference_stiffness_tetrad_3d = reference_stiffness_tetrad;
  else if constexpr(dim == 2)
  {
    for (unsigned int i = 0; i < 3; i++)
      for (unsigned int j = 0; j < 3; j++)
        for (unsigned int k = 0; k < 3; k++)
          for (unsigned int l = 0; l < 3; l++)
            if (i == j && j == k && k == l)
              reference_stiffness_tetrad_3d[i][j][k][l] =
                C1111 / characteristic_stiffness;
            else if (i == k && j == l)
              reference_stiffness_tetrad_3d[i][j][k][l] =
                C1212 / characteristic_stiffness;
            else if (i == j && k == l)
              reference_stiffness_tetrad_3d[i][j][k][l] =
                C1122 / characteristic_stiffness;
  }
  else
    Assert(false, dealii::ExcNotImplemented());

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
          dealii::SymmetricTensor<4,dim>  stiffness_tetrad;

          dealii::Tensor<2,dim> rotation_tensor =
            crystals_data->get_rotation_tensor(crystal_id);

          dealii::SymmetricTensor<4,3>    stiffness_tetrad_3d;

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

          if constexpr(dim == 3)
            stiffness_tetrad_3d = stiffness_tetrad;
          else if constexpr(dim == 2)
          {
            dealii::Tensor<2,3>   rotation_tensor_3d =
              crystals_data->get_3d_rotation_tensor(crystal_id);

            for (unsigned int i = 0; i < 3; i++)
              for (unsigned int j = i; j < 3; j++)
                for (unsigned int k = 0; k < 3; k++)
                  for (unsigned int l = k; l < 3; l++)
                    for (unsigned int o = 0; o < 3; o++)
                      for (unsigned int p = 0; p < 3; p++)
                        for (unsigned int q = 0; q < 3; q++)
                          for (unsigned int r = 0; r < 3; r++)
                            stiffness_tetrad_3d[i][j][k][l] +=
                              rotation_tensor_3d[i][o] *
                              rotation_tensor_3d[j][p] *
                              rotation_tensor_3d[k][q] *
                              rotation_tensor_3d[l][r] *
                              reference_stiffness_tetrad_3d[o][p][q][r];
          }
          else
            Assert(false, dealii::ExcNotImplemented());

          stiffness_tetrads.push_back(stiffness_tetrad);
          stiffness_tetrads_3d.push_back(stiffness_tetrad_3d);
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
ScalarMicrostressLaw<dim>::ScalarMicrostressLaw(
  const std::shared_ptr<CrystalsData<dim>>                &crystals_data,
  const RunTimeParameters::ScalarMicrostressLawParameters parameters,
  const RunTimeParameters::HardeningLaw                   hardening_law_prm,
  const double characteristic_slip_resistance)
:
crystals_data(crystals_data),
regularization_function(parameters.regularization_function),
regularization_parameter(parameters.regularization_parameter),
linear_hardening_modulus(hardening_law_prm.linear_hardening_modulus),
hardening_parameter(hardening_law_prm.hardening_parameter),
characteristic_slip_resistance(characteristic_slip_resistance),
flag_perfect_plasticity(hardening_law_prm.flag_perfect_plasticity),
flag_rate_independent(parameters.flag_rate_independent)
{}



template<int dim>
double ScalarMicrostressLaw<dim>::get_scalar_microstress(
  const double slip_value,
  const double old_slip_value,
  const double slip_resistance,
  const double time_step_size)
{
  AssertThrow(crystals_data->is_initialized(),
              dealii::ExcMessage("The underlying CrystalsData<dim>"
                                  " instance has not been "
                                  " initialized."));

  AssertIsFinite(slip_value);
  AssertIsFinite(old_slip_value);
  AssertIsFinite(slip_resistance);
  AssertIsFinite(time_step_size);

  const double slip_rate =
    (slip_value - old_slip_value) / time_step_size;

  if (flag_rate_independent)
  {
    return (slip_resistance * sgn(slip_rate));
  }
  else
  {
    const double regularization_function_value =
      get_regularization_function_value(slip_rate);

    AssertIsFinite(regularization_function_value);

    return (slip_resistance * regularization_function_value);
  }
}



template<int dim>
dealii::FullMatrix<double> ScalarMicrostressLaw<dim>::
  get_jacobian(
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

  dealii::FullMatrix<double> jacobian(crystals_data->get_n_slips());

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
  {
    for (unsigned int slip_id_beta = 0;
        slip_id_beta < crystals_data->get_n_slips();
        ++slip_id_beta)
    {
      jacobian[slip_id_alpha][slip_id_beta] = 0.;

      if (!flag_perfect_plasticity)
      {
        jacobian[slip_id_alpha][slip_id_beta] =
          get_hardening_matrix_entry(slip_id_alpha == slip_id_beta);

        if (flag_rate_independent)
        {
          jacobian[slip_id_alpha][slip_id_beta] *=
            sgn(compute_slip_rate(q_point, slip_id_alpha)) *
            sgn(compute_slip_rate(q_point, slip_id_beta));
        }
        else
        {
          // The latter term ought to be a sgn function but for the sake
          // of symmetry it is also approximated by a sigmoid function
          jacobian[slip_id_alpha][slip_id_beta] *=
            get_regularization_function_value(
              compute_slip_rate(q_point, slip_id_alpha)) *
            get_regularization_function_value(
              compute_slip_rate(q_point, slip_id_beta));
        }
      }

      if (slip_id_alpha == slip_id_beta &&
          !flag_rate_independent)
      {
        jacobian[slip_id_alpha][slip_id_beta] +=
          (slip_resistances[slip_id_alpha] / time_step_size *
            get_regularization_function_derivative_value(
              compute_slip_rate(q_point, slip_id_alpha)));
      }

      AssertIsFinite(jacobian[slip_id_alpha][slip_id_beta]);
    }
  }

  return jacobian;
}



template <int dim>
double ScalarMicrostressLaw<dim>::
get_regularization_function_value(const double slip_rate) const
{
  AssertIsFinite(slip_rate);

  double regularization_function_value = 0.0;

  switch (regularization_function)
  {
  case RunTimeParameters::RegularizationFunction::Atan:
    {
      regularization_function_value =
        2.0 / M_PI * std::atan(
          M_PI / 2.0 * slip_rate / regularization_parameter);
    }
    break;
  case RunTimeParameters::RegularizationFunction::Sqrt:
    {
      regularization_function_value =
        slip_rate / std::sqrt(slip_rate * slip_rate +
                              regularization_parameter *
                              regularization_parameter);
    }
    break;
  case RunTimeParameters::RegularizationFunction::Gd:
    {
      regularization_function_value =
        2.0 / M_PI * std::atan(std::sinh(
          M_PI / 2.0 * slip_rate / regularization_parameter));
    }
    break;
  case RunTimeParameters::RegularizationFunction::Tanh:
    {
      regularization_function_value =
        std::tanh(slip_rate / regularization_parameter);
    }
    break;
  case RunTimeParameters::RegularizationFunction::Erf:
    {
      regularization_function_value =
        std::erf(std::sqrt(M_PI) / 2.0 * slip_rate /
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

  AssertIsFinite(regularization_function_value);

  return regularization_function_value;
}



template <int dim>
double ScalarMicrostressLaw<dim>::
get_regularization_function_derivative_value(const double slip_rate) const
{
  AssertIsFinite(slip_rate);

  double regularization_function_derivative_value = 0.0;

  switch (regularization_function)
  {
  case RunTimeParameters::RegularizationFunction::Atan:
    {
      regularization_function_derivative_value =
        regularization_parameter /
        (regularization_parameter *
         regularization_parameter +
         M_PI * M_PI * slip_rate * slip_rate / 4.);
    }
    break;
  case RunTimeParameters::RegularizationFunction::Sqrt:
    {
      regularization_function_derivative_value =
        regularization_parameter *
        regularization_parameter /
        std::pow(slip_rate * slip_rate +
                  regularization_parameter *
                  regularization_parameter,
                 1.5);
    }
    break;
  case RunTimeParameters::RegularizationFunction::Gd:
    {
      regularization_function_derivative_value =
        1.0 / std::cosh(M_PI / 2. * slip_rate /
        regularization_parameter) /
        regularization_parameter;
    }
    break;
  case RunTimeParameters::RegularizationFunction::Tanh:
    {
      regularization_function_derivative_value =
        std::pow(1.0 / std::cosh(slip_rate / regularization_parameter),
                 2) / regularization_parameter;
    }
    break;
  case RunTimeParameters::RegularizationFunction::Erf:
    {
      regularization_function_derivative_value =
        1. / regularization_parameter *
        std::exp(-M_PI * slip_rate * slip_rate /
                 regularization_parameter /
                 regularization_parameter / 4.);
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

  AssertIsFinite(regularization_function_derivative_value);

  return regularization_function_derivative_value;
}



template<int dim>
VectorialMicrostressLaw<dim>::VectorialMicrostressLaw(
  const std::shared_ptr<CrystalsData<dim>> &crystals_data,
  const RunTimeParameters::VectorialMicrostressLawParameters parameters)
:
crystals_data(crystals_data),
energetic_length_scale(parameters.energetic_length_scale),
initial_slip_resistance(parameters.initial_slip_resistance),
defect_energy_index(parameters.defect_energy_index),
flag_init_was_called(false)
{}



template<int dim>
void VectorialMicrostressLaw<dim>::init(
  const bool flag_dimensionless_formulation)
{
  AssertThrow(crystals_data->is_initialized(),
              dealii::ExcMessage("The underlying CrystalsData<dim>"
                                  " instance has not been "
                                  " initialized."));

  factor = flag_dimensionless_formulation ?
              1.0 :
              initial_slip_resistance *
                std::pow(energetic_length_scale, defect_energy_index);

  for (unsigned int crystal_id = 0;
        crystal_id < crystals_data->get_n_crystals();
        crystal_id++)
  {
    std::vector<dealii::SymmetricTensor<2,dim>>
      slip_direction_dyads_per_crystal;

    std::vector<dealii::SymmetricTensor<2,dim>>
      slip_binormal_dyads_per_crystal;

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

      for (unsigned int i = 0;
           i < slip_direction_outer_product.n_independent_components;
           ++i)
      {
        AssertIsFinite(slip_direction_outer_product.access_raw_entry(i));
        AssertIsFinite(slip_orthogonal_outer_product.access_raw_entry(i));
      }

      slip_direction_dyads_per_crystal.push_back(
        slip_direction_outer_product);

      slip_binormal_dyads_per_crystal.push_back(
        slip_orthogonal_outer_product);
    }

    slip_direction_dyads.push_back(
      slip_direction_dyads_per_crystal);

    slip_binormal_dyads.push_back(
      slip_binormal_dyads_per_crystal);
  }

  Assert(
    slip_direction_dyads.size() ==
      crystals_data->get_n_crystals(),
    dealii::ExcNotImplemented());

  Assert(
    slip_binormal_dyads.size() ==
      crystals_data->get_n_crystals(),
    dealii::ExcNotImplemented());

  flag_init_was_called = true;
}



template <int dim>
dealii::Tensor<1,dim> VectorialMicrostressLaw<dim>::
get_vectorial_microstress(
  const unsigned int          crystal_id,
  const unsigned int          slip_id,
  const dealii::Tensor<1,dim> slip_gradient) const
{
  AssertIndexRange(crystal_id, crystals_data->get_n_crystals());
  AssertIndexRange(slip_id, crystals_data->get_n_slips());

  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The VectorialMicrostressLaw<dim> "
                                 "instance has not been initialized."));

  return (
    factor *
    (
      std::pow(
        std::abs(-crystals_data->get_slip_direction(crystal_id, slip_id) *
                slip_gradient),
        defect_energy_index - 2.) *
      slip_direction_dyads[crystal_id][slip_id]
      +
      std::pow(
        std::abs(crystals_data->get_slip_orthogonal(crystal_id, slip_id) *
                slip_gradient),
        defect_energy_index - 2.) *
      slip_binormal_dyads[crystal_id][slip_id]
    ) *
    slip_gradient);
}



template <int dim>
dealii::SymmetricTensor<2,dim> VectorialMicrostressLaw<dim>::
get_jacobian(
  const unsigned int          crystal_id,
  const unsigned int          slip_id,
  const dealii::Tensor<1,dim> slip_gradient) const
{
  AssertIndexRange(crystal_id, crystals_data->get_n_crystals());
  AssertIndexRange(slip_id, crystals_data->get_n_slips());

  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The VectorialMicrostressLaw<dim> "
                                 "instance has not been initialized."));

  return (
    factor *
    (defect_energy_index - 1.0) *
    (
      std::pow(
        std::abs(-crystals_data->get_slip_direction(crystal_id, slip_id) *
                slip_gradient),
        defect_energy_index - 2.) *
      slip_direction_dyads[crystal_id][slip_id]
      +
      std::pow(
        std::abs(crystals_data->get_slip_orthogonal(crystal_id, slip_id) *
                slip_gradient),
        defect_energy_index - 2.) *
      slip_binormal_dyads[crystal_id][slip_id]
    ));
}



template<int dim>
MicrotractionLaw<dim>::MicrotractionLaw(
  const std::shared_ptr<CrystalsData<dim>> &crystals_data,
  const RunTimeParameters::MicrotractionLawParameters &parameters,
  const double characteristic_vectorial_microstress)
:
crystals_data(crystals_data),
grain_boundary_modulus(parameters.grain_boundary_modulus),
characteristic_vectorial_microstress(
  characteristic_vectorial_microstress)
{}



template<>
MicrotractionLaw<2>::GrainInteractionModuli
MicrotractionLaw<2>::get_grain_interaction_moduli(
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
            std::fabs(intra_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]) <= (1.0 + 1e-14),
          dealii::ExcMessage(
            "The interaction moduli should be inside the "
            "range [0,1]. Its value is " +
            std::to_string(
              std::fabs(
                intra_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]))));

        AssertThrow(
          std::fabs(inter_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]) >= 0.0 &&
            std::fabs(inter_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]) <= (1.0 + 1e-14),
          dealii::ExcMessage(
            "The interaction moduli should be inside the "
            "range [0,1].  Its value is " +
            std::to_string(
              std::fabs(
                inter_grain_interaction_moduli_per_q_point[slip_id_alpha][slip_id_beta]))));

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
MicrotractionLaw<3>::GrainInteractionModuli
MicrotractionLaw<3>::get_grain_interaction_moduli(
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
double MicrotractionLaw<dim>::get_microtraction(
  const unsigned int                      q_point,
  const unsigned int                      slip_id_alpha,
  const GrainInteractionModuli            grain_interaction_moduli,
  const std::vector<std::vector<double>>  slip_values_current_cell,
  const std::vector<std::vector<double>>  slip_values_neighbour_cell) const
{
  double microtraction = 0.0;

  for (unsigned int slip_id_beta = 0;
       slip_id_beta < crystals_data->get_n_slips(); slip_id_beta++)
  {
    microtraction +=
      grain_interaction_moduli.first[q_point][slip_id_alpha][slip_id_beta] *
      slip_values_current_cell[slip_id_beta][q_point]
      -
      grain_interaction_moduli.second[q_point][slip_id_alpha][slip_id_beta] *
      slip_values_neighbour_cell[slip_id_beta][q_point];
  }

  microtraction *= -grain_boundary_modulus /
                      characteristic_vectorial_microstress;

  return (microtraction);
}



template <int dim>
const dealii::FullMatrix<double>
MicrotractionLaw<dim>::
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

  intra_gateaux_derivative *= -grain_boundary_modulus /
                                  characteristic_vectorial_microstress;

  return (intra_gateaux_derivative);
}



template <int dim>
const dealii::FullMatrix<double>
MicrotractionLaw<dim>::
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

  inter_gateaux_derivative *= grain_boundary_modulus /
                                characteristic_vectorial_microstress;

  return (inter_gateaux_derivative);
}



template <>
double MicrotractionLaw<2>::get_free_energy_density(
  const unsigned int                      neighbor_cell_crystal_id,
  const unsigned int                      current_cell_crystal_id,
  const unsigned int                      quadrature_point_id,
  std::vector<dealii::Tensor<1,2>>        normal_vector_values,
  const std::vector<std::vector<double>>  neighbor_cell_slip_values,
  const std::vector<std::vector<double>>  current_cell_slip_values) const
{
  std::vector<dealii::Tensor<1,2>> neighbor_cell_slip_directions =
    crystals_data->get_slip_directions(neighbor_cell_crystal_id);

  std::vector<dealii::Tensor<1,2>> neighbor_cell_slip_normals =
    crystals_data->get_slip_normals(neighbor_cell_crystal_id);

  std::vector<dealii::Tensor<1,2>> current_cell_slip_directions =
    crystals_data->get_slip_directions(current_cell_crystal_id);

  std::vector<dealii::Tensor<1,2>> current_cell_slip_normals =
    crystals_data->get_slip_normals(current_cell_crystal_id);

  dealii::Tensor<1,2> burgers_tensor;

  burgers_tensor = 0.;

  for (unsigned int slip_id = 0;
       slip_id < crystals_data->get_n_slips(); ++slip_id)
  {
    burgers_tensor +=
      neighbor_cell_slip_values[slip_id][quadrature_point_id] *
      neighbor_cell_slip_directions[slip_id] *
      (neighbor_cell_slip_normals[slip_id][0] *
       normal_vector_values[quadrature_point_id][1]
       -
       neighbor_cell_slip_normals[slip_id][1] *
       normal_vector_values[quadrature_point_id][0])
      -
      current_cell_slip_values[slip_id][quadrature_point_id] *
      current_cell_slip_directions[slip_id] *
      (current_cell_slip_normals[slip_id][0] *
       normal_vector_values[quadrature_point_id][1]
       -
       current_cell_slip_normals[slip_id][1] *
       normal_vector_values[quadrature_point_id][0]);
  }

  for (unsigned int i = 0; i < 2; ++i)
      AssertIsFinite(burgers_tensor[i]);

  const double free_energy_density =
    0.5 * grain_boundary_modulus *
    dealii::scalar_product(burgers_tensor, burgers_tensor);

  AssertIsFinite(free_energy_density);

  return (free_energy_density);
}



template <>
double MicrotractionLaw<3>::get_free_energy_density(
  const unsigned int                      neighbor_cell_crystal_id,
  const unsigned int                      current_cell_crystal_id,
  const unsigned int                      quadrature_point_id,
  std::vector<dealii::Tensor<1,3>>      normal_vector_values,
  const std::vector<std::vector<double>>  neighbor_cell_slip_values,
  const std::vector<std::vector<double>>  current_cell_slip_values) const
{
  std::vector<dealii::Tensor<1,3>> neighbor_cell_slip_directions =
    crystals_data->get_slip_directions(neighbor_cell_crystal_id);

  std::vector<dealii::Tensor<1,3>> neighbor_cell_slip_normals =
    crystals_data->get_slip_normals(neighbor_cell_crystal_id);

  std::vector<dealii::Tensor<1,3>> current_cell_slip_directions =
    crystals_data->get_slip_directions(current_cell_crystal_id);

  std::vector<dealii::Tensor<1,3>> current_cell_slip_normals =
    crystals_data->get_slip_normals(current_cell_crystal_id);

  dealii::Tensor<2,3> burgers_tensor;

  burgers_tensor = 0.;

  for (unsigned int slip_id = 0;
       slip_id < crystals_data->get_n_slips(); ++slip_id)
  {
    burgers_tensor +=
      neighbor_cell_slip_values[slip_id][quadrature_point_id] *
      dealii::outer_product(
        neighbor_cell_slip_directions[slip_id],
        dealii::cross_product_3d(
          neighbor_cell_slip_normals[slip_id],
          normal_vector_values[quadrature_point_id]))
      -
      current_cell_slip_values[slip_id][quadrature_point_id] *
      dealii::outer_product(
        current_cell_slip_directions[slip_id],
        dealii::cross_product_3d(
          current_cell_slip_normals[slip_id],
          normal_vector_values[quadrature_point_id]));
  }

  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      AssertIsFinite(burgers_tensor[i][j]);

  const double free_energy_density =
    0.5 * grain_boundary_modulus *
    dealii::scalar_product(burgers_tensor, burgers_tensor);

  AssertIsFinite(free_energy_density);

  return (free_energy_density);
}



template<int dim>
CohesiveLaw<dim>::CohesiveLaw(
  const RunTimeParameters::CohesiveLawParameters parameters,
  const double characteristic_stress,
  const double characteristic_displacement)
:
critical_cohesive_traction(parameters.critical_cohesive_traction),
critical_opening_displacement(parameters.critical_opening_displacement),
tangential_to_normal_stiffness_ratio(parameters.tangential_to_normal_stiffness_ratio),
characteristic_stress(characteristic_stress),
characteristic_displacement(characteristic_displacement)
{}



template <int dim>
dealii::Tensor<1,dim>
CohesiveLaw<dim>::get_cohesive_traction(
  const dealii::Tensor<1,dim> opening_displacement,
  const dealii::Tensor<1,dim> normal_vector,
  const double                max_effective_opening_displacement,
  const double                old_effective_opening_displacement,
  const double                time_step_size) const
{
  AssertIsFinite(max_effective_opening_displacement);
  AssertIsFinite(old_effective_opening_displacement);
  AssertIsFinite(1.0 / time_step_size);

  // Initiate cohesive traction
  dealii::Tensor<1,dim> cohesive_traction;

  // Get the effective opening displacement, the effective
  // direction and the effective identity tensor
  const EffectiveQuantities effective_quantities =
    get_effective_quantities(opening_displacement, normal_vector);

  Assert(
    effective_quantities.opening_displacement <=
      max_effective_opening_displacement,
    dealii::ExcMessage(
      "The effective opening displacement is not suppose to be "
      "bigger than the maximum. An update_values() call ought to "
      "be missing in code."));

  // Compute the effective opening displacement rate
  const double effective_opening_displacement_rate =
    (effective_quantities.opening_displacement -
     old_effective_opening_displacement) /
    time_step_size;

  // The cohesive traction has the same direction as the
  // effective direction
  cohesive_traction = effective_quantities.direction;

  // Loading behavior
  if (effective_quantities.opening_displacement ==
        max_effective_opening_displacement &&
      effective_opening_displacement_rate >= 0.0)
  {
    cohesive_traction *=
      get_effective_cohesive_traction(
        characteristic_displacement *
        effective_quantities.opening_displacement);
  }
  // Unloading and reloading behavior
  else if (effective_quantities.opening_displacement <
             max_effective_opening_displacement ||
           (effective_quantities.opening_displacement ==
              max_effective_opening_displacement &&
            effective_opening_displacement_rate < 0.0))
  {
    cohesive_traction *=
      get_effective_cohesive_traction(
        characteristic_displacement *
        max_effective_opening_displacement) *
      effective_quantities.opening_displacement /
      max_effective_opening_displacement;
  }
  else
  {
    Assert(false, dealii::ExcInternalError());
  }

  for (unsigned int i = 0; i < dim; ++i)
  {
    AssertIsFinite(cohesive_traction[i]);
  }

  return (cohesive_traction / characteristic_stress);
}



template <int dim>
dealii::SymmetricTensor<2,dim>
CohesiveLaw<dim>::get_jacobian(
  const dealii::Tensor<1,dim> opening_displacement,
  const dealii::Tensor<1,dim> normal_vector,
  const double                max_effective_opening_displacement,
  const double                old_effective_opening_displacement,
  const double                time_step_size) const
{
  AssertIsFinite(max_effective_opening_displacement);
  AssertIsFinite(old_effective_opening_displacement);
  AssertIsFinite(1.0 / time_step_size);

  // Initiate Gateaux derivative
  dealii::SymmetricTensor<2,dim> jacobian;

  // Get the effective opening displacement, the effective
  // direction and the effective identity tensor
  const EffectiveQuantities effective_quantities =
    get_effective_quantities(opening_displacement, normal_vector);

  // Compute the effective opening displacement rate
  const double effective_opening_displacement_rate =
    (effective_quantities.opening_displacement -
     old_effective_opening_displacement) /
    time_step_size;

  Assert(
    effective_quantities.opening_displacement <=
      max_effective_opening_displacement,
    dealii::ExcMessage(
      "The effective opening displacement is not suppose to be "
      "bigger than the maximum. An update_values() call ought to "
      "be missing in code."));

  // Loading behavior
  if (effective_quantities.opening_displacement ==
        max_effective_opening_displacement &&
      effective_opening_displacement_rate >= 0.0)
  {
    jacobian =
      critical_cohesive_traction /
      critical_opening_displacement *
      std::exp(1.0 - characteristic_displacement *
        effective_quantities.opening_displacement /
                     critical_opening_displacement) *
      (effective_quantities.identity_tensor
       -
       characteristic_displacement *
       effective_quantities.opening_displacement /
       critical_opening_displacement *
       dealii::symmetrize(dealii::outer_product(
        effective_quantities.direction,
        effective_quantities.direction)));
  }
  // Unloading and reloading behavior
  else if (effective_quantities.opening_displacement <
             max_effective_opening_displacement ||
           (effective_quantities.opening_displacement ==
              max_effective_opening_displacement &&
            effective_opening_displacement_rate < 0.0))
  {
    jacobian =
      get_effective_cohesive_traction(
        characteristic_displacement*
        max_effective_opening_displacement) /
      characteristic_displacement /
      max_effective_opening_displacement *
      effective_quantities.identity_tensor;
  }
  else
    Assert(false, dealii::ExcInternalError());

  for (unsigned int i = 0;
       i < jacobian.n_independent_components; ++i)
    AssertIsFinite(jacobian.access_raw_entry(i));

  return (jacobian / characteristic_stress);
}



template <int dim>
double CohesiveLaw<dim>::get_free_energy_density(
  const double effective_opening_displacement) const
{
  AssertIsFinite(effective_opening_displacement);

  const double free_energy_density =
    critical_cohesive_traction *
    critical_opening_displacement *
    std::exp(1.0) *
    (1.0 -
    (1.0 +
      effective_opening_displacement /
      critical_opening_displacement) *
     std::exp(
      -effective_opening_displacement /
      critical_opening_displacement));

  AssertIsFinite(free_energy_density);

  return (free_energy_density);
}



template <int dim>
double CohesiveLaw<dim>::get_effective_opening_displacement(
  const dealii::Tensor<1,dim> opening_displacement,
  const dealii::Tensor<1,dim> normal_vector) const
{
  // Compute projectors
  dealii::SymmetricTensor<2,dim> normal_projector =
    dealii::symmetrize(dealii::outer_product(normal_vector,
                                              normal_vector));

  dealii::SymmetricTensor<2,dim> tangential_projector =
    dealii::unit_symmetric_tensor<dim>() - normal_projector;

  // Decompose the opening displacement
  double normal_opening_displacement =
    normal_vector * opening_displacement;

  double tangential_opening_displacement =
    (tangential_projector * opening_displacement).norm();

  const double effective_opening_displacement =
    std::sqrt(macaulay_brackets(normal_opening_displacement) *
              macaulay_brackets(normal_opening_displacement)
              +
              tangential_to_normal_stiffness_ratio *
              tangential_to_normal_stiffness_ratio *
              tangential_opening_displacement *
              tangential_opening_displacement);

  AssertIsFinite(effective_opening_displacement);

  return (effective_opening_displacement);
}



template <int dim>
typename CohesiveLaw<dim>::EffectiveQuantities
CohesiveLaw<dim>::get_effective_quantities(
  const dealii::Tensor<1,dim> opening_displacement,
  const dealii::Tensor<1,dim> normal_vector) const
{
  // Compute projectors
  const dealii::SymmetricTensor<2,dim> normal_projector =
    dealii::symmetrize(dealii::outer_product(normal_vector,
                                             normal_vector));

  const dealii::SymmetricTensor<2,dim> tangential_projector =
    dealii::unit_symmetric_tensor<dim>() - normal_projector;

  // Decompose the opening displacement
  const double normal_opening_displacement =
    normal_vector * opening_displacement;

  const dealii::Tensor<1,dim> tangential_opening_displacement =
    tangential_projector * opening_displacement;

  const double tangential_opening_displacement_norm =
    tangential_opening_displacement.norm();

  // Compute effective quantities
  double effective_opening_displacement =
    std::sqrt(macaulay_brackets(normal_opening_displacement) *
              macaulay_brackets(normal_opening_displacement)
              +
              tangential_to_normal_stiffness_ratio *
              tangential_to_normal_stiffness_ratio *
              tangential_opening_displacement_norm *
              tangential_opening_displacement_norm);

  dealii::Tensor<1,dim> effective_direction =
    macaulay_brackets(normal_opening_displacement) * normal_vector
    +
    tangential_to_normal_stiffness_ratio *
    tangential_to_normal_stiffness_ratio *
    tangential_opening_displacement;

  if (effective_opening_displacement != 0.0)
    effective_direction /= effective_opening_displacement;

  const dealii::SymmetricTensor<2,dim> effective_identity_tensor =
    macaulay_brackets(
      normal_opening_displacement /
      std::abs(normal_opening_displacement)) *
    normal_projector
    +
    tangential_to_normal_stiffness_ratio *
    tangential_to_normal_stiffness_ratio *
    tangential_projector;

  AssertIsFinite(normal_opening_displacement);
  AssertIsFinite(effective_opening_displacement);

  return EffectiveQuantities(
    effective_opening_displacement,
    effective_direction,
    effective_identity_tensor,
    normal_opening_displacement);
}



DegradationFunction::DegradationFunction(
    const RunTimeParameters::DegradationFunction parameters)
:
degradation_exponent(parameters.degradation_exponent)
{}



template<int dim>
ContactLaw<dim>::ContactLaw(
  const RunTimeParameters::ContactLawParameters parameters,
  const double characteristic_stress,
  const double characteristic_displacement)
:
penalty_coefficient(parameters.penalty_coefficient),
characteristic_stress(characteristic_stress),
characteristic_displacement(characteristic_displacement)
{}



template <int dim>
dealii::Tensor<1,dim>
ContactLaw<dim>::get_contact_traction(
  const dealii::Tensor<1,dim> opening_displacement,
  const dealii::Tensor<1,dim> normal_vector) const
{
  // Initiate cohesive traction.
  dealii::Tensor<1,dim> contact_traction;

  // Compute normal component of the opening displacement
  const double normal_opening_displacement =
    opening_displacement * normal_vector;

  // Compute cohesive traction
  contact_traction -=
    penalty_coefficient /
    characteristic_stress *
    characteristic_displacement *
    macaulay_brackets(-normal_opening_displacement) * normal_vector;

  for (unsigned int i = 0; i < dim; ++i)
  {
    AssertIsFinite(contact_traction[i]);
  }

  return (contact_traction);
}



template <int dim>
dealii::SymmetricTensor<2,dim>
ContactLaw<dim>::get_jacobian(
  const dealii::Tensor<1,dim> opening_displacement,
  const dealii::Tensor<1,dim> normal_vector) const
{
  // Initiate Gateaux derivative
  dealii::SymmetricTensor<2,dim> jacobian;

  // Compute normal component of the opening displacement
  const double normal_opening_displacement =
    opening_displacement * normal_vector;

  // Compute Gateaux derivatice for the neighbor cell
  jacobian =
    penalty_coefficient /
    characteristic_stress *
    macaulay_brackets(-normal_opening_displacement /
                      std::abs(normal_opening_displacement)) *
    dealii::symmetrize(dealii::outer_product(normal_vector,
                                             normal_vector));

  for (unsigned int i = 0;
       i < jacobian.n_independent_components; ++i)
  {
    AssertIsFinite(jacobian.access_raw_entry(i));
  }

  return (jacobian);
}


} // ConstitutiveLaws



} // gCP


template class gCP::Kinematics::ElasticStrain<2>;
template class gCP::Kinematics::ElasticStrain<3>;

template class gCP::ConstitutiveLaws::HookeLaw<2>;
template class gCP::ConstitutiveLaws::HookeLaw<3>;

template class gCP::ConstitutiveLaws::ResolvedShearStressLaw<2>;
template class gCP::ConstitutiveLaws::ResolvedShearStressLaw<3>;

template class gCP::ConstitutiveLaws::ScalarMicrostressLaw<2>;
template class gCP::ConstitutiveLaws::ScalarMicrostressLaw<3>;

template class gCP::ConstitutiveLaws::VectorialMicrostressLaw<2>;
template class gCP::ConstitutiveLaws::VectorialMicrostressLaw<3>;

template class gCP::ConstitutiveLaws::MicrotractionLaw<2>;
template class gCP::ConstitutiveLaws::MicrotractionLaw<3>;

template class gCP::ConstitutiveLaws::CohesiveLaw<2>;
template class gCP::ConstitutiveLaws::CohesiveLaw<3>;

template class gCP::ConstitutiveLaws::ContactLaw<2>;
template class gCP::ConstitutiveLaws::ContactLaw<3>;
