#ifndef INCLUDE_CONSTITUTIVE_EQUATIONS_H_
#define INCLUDE_CONSTITUTIVE_EQUATIONS_H_

#include <gCP/crystal_data.h>
#include <gCP/quadrature_point_history.h>
#include <gCP/run_time_parameters.h>

#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/generic_linear_algebra.h>



namespace gCP
{



namespace Kinematics
{


template <int dim>
class ElasticStrain
{
public:
  ElasticStrain(std::shared_ptr<CrystalsData<dim>>  crystals_data);

  const dealii::SymmetricTensor<2,dim> get_elastic_strain_tensor(
    const unsigned int                      crystal_id,
    const unsigned int                      q_point,
    const dealii::SymmetricTensor<2,dim>    strain_tensor_value,
    const std::vector<std::vector<double>>  slip_values) const;

  const dealii::SymmetricTensor<2,dim> get_plastic_strain_tensor(
    const unsigned int                      crystal_id,
    const unsigned int                      q_point,
    const std::vector<std::vector<double>>  slip_values) const;

private:
  std::shared_ptr<const CrystalsData<dim>>    crystals_data;
};



} // namespace Kinematics


namespace ConstitutiveLaws
{



template<int dim>
class HookeLaw
{
public:
  HookeLaw(const RunTimeParameters::HookeLawParameters  parameters);

  HookeLaw(const std::shared_ptr<CrystalsData<dim>>     &crystals_data,
           const RunTimeParameters::HookeLawParameters  parameters);

  void init();

  const dealii::SymmetricTensor<4,dim> &get_stiffness_tetrad() const;

  const dealii::SymmetricTensor<4,dim> &get_stiffness_tetrad(
    const unsigned int crystal_id) const;

  const dealii::SymmetricTensor<4,3> &get_stiffness_tetrad_3d() const;

  const dealii::SymmetricTensor<4,3> &get_stiffness_tetrad_3d(
    const unsigned int crystal_id) const;

  const dealii::SymmetricTensor<2,dim> get_stress_tensor(
    const dealii::SymmetricTensor<2,dim> strain_tensor_values) const;

  const dealii::SymmetricTensor<2,dim> get_stress_tensor(
    const unsigned int                    crystal_id,
    const dealii::SymmetricTensor<2,dim>  strain_tensor_values) const;

private:
  enum class Crystallite
  {
    Monocrystalline,
    Polycrystalline
  };

  std::shared_ptr<const CrystalsData<dim>>    crystals_data;

  Crystallite                                 crystallite;

  const double                                C1111;

  const double                                C1122;

  const double                                C1212;

  dealii::SymmetricTensor<4,dim>              reference_stiffness_tetrad;

  std::vector<dealii::SymmetricTensor<4,dim>> stiffness_tetrads;

  dealii::SymmetricTensor<4,3>                reference_stiffness_tetrad_3d;

  std::vector<dealii::SymmetricTensor<4,3>>   stiffness_tetrads_3d;


  bool                                        flag_init_was_called;
};



template <int dim>
inline const dealii::SymmetricTensor<4,dim>
&HookeLaw<dim>::get_stiffness_tetrad() const
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

  return (reference_stiffness_tetrad);
}



template <int dim>
inline const dealii::SymmetricTensor<4,dim>
&HookeLaw<dim>::get_stiffness_tetrad(const unsigned int crystal_id) const
{
  AssertThrow(crystallite == Crystallite::Polycrystalline,
              dealii::ExcMessage("This overload is meant for the"
                                 " case of a polycrystalline."
                                 " Nonetheless no CrystalsData<dim>'s"
                                 " shared pointer was passed on to the"
                                 " constructor"));

  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The HookeLaw<dim> instance has not"
                                 " been initialized."));

  AssertIndexRange(crystal_id, crystals_data->get_n_crystals());

  return (stiffness_tetrads[crystal_id]);
}



template <int dim>
inline const dealii::SymmetricTensor<4,3>
&HookeLaw<dim>::get_stiffness_tetrad_3d() const
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

  return (reference_stiffness_tetrad_3d);
}



template <int dim>
inline const dealii::SymmetricTensor<4,3>
&HookeLaw<dim>::get_stiffness_tetrad_3d(const unsigned int crystal_id) const
{
  AssertThrow(crystallite == Crystallite::Polycrystalline,
              dealii::ExcMessage("This overload is meant for the"
                                 " case of a polycrystalline."
                                 " Nonetheless no CrystalsData<dim>'s"
                                 " shared pointer was passed on to the"
                                 " constructor"));

  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The HookeLaw<dim> instance has not"
                                 " been initialized."));

  AssertIndexRange(crystal_id, crystals_data->get_n_crystals());

  return (stiffness_tetrads_3d[crystal_id]);
}


template<int dim>
class ResolvedShearStressLaw
{
public:
  ResolvedShearStressLaw(
    const std::shared_ptr<CrystalsData<dim>>  &crystals_data);

  double get_resolved_shear_stress(
    const unsigned int                    crystal_id,
    const unsigned int                    slip_id,
    const dealii::SymmetricTensor<2,dim>  stress_tensor) const;

private:
  std::shared_ptr<const CrystalsData<dim>>  crystals_data;
};



template <int dim>
inline double
ResolvedShearStressLaw<dim>::get_resolved_shear_stress(
  const unsigned int                    crystal_id,
  const unsigned int                    slip_id,
  const dealii::SymmetricTensor<2,dim>  stress_tensor) const
{
  AssertThrow(crystals_data->is_initialized(),
              dealii::ExcMessage("The underlying CrystalsData<dim>"
                                  " instance has not been "
                                  " initialized."));

  AssertIndexRange(crystal_id, crystals_data->get_n_crystals());
  AssertIndexRange(slip_id, crystals_data->get_n_slips());

  return (stress_tensor *
          crystals_data->get_symmetrized_schmid_tensor(crystal_id,
                                                       slip_id));
}



template<int dim>
class ScalarMicroscopicStressLaw
{
public:
  ScalarMicroscopicStressLaw(
    const std::shared_ptr<CrystalsData<dim>>                      &crystals_data,
    const RunTimeParameters::ScalarMicroscopicStressLawParameters parameters);

  double get_scalar_microscopic_stress(
    const double slip_value,
    const double old_slip_value,
    const double slip_resistance,
    const double time_step_size);

  dealii::FullMatrix<double> get_gateaux_derivative_matrix(
    const unsigned int                      q_point,
    const std::vector<std::vector<double>>  slip_values,
    const std::vector<std::vector<double>>  old_slip_values,
    const std::vector<double>               slip_resistances,
    const double                            time_step_size);


private:
  std::shared_ptr<const CrystalsData<dim>>  crystals_data;

  RunTimeParameters::RegularizationFunction regularization_function;

  const double                              regularization_parameter;

  const double                              initial_slip_resistance;

  const double                              linear_hardening_modulus;

  const double                              hardening_parameter;

  double get_hardening_matrix_entry(const bool self_hardening) const;

  double sgn(const double value) const;

  double get_regularization_factor(const double slip_rate) const;
};



template <int dim>
inline double
ScalarMicroscopicStressLaw<dim>::get_hardening_matrix_entry(
  const bool self_hardening) const
{
  return (linear_hardening_modulus *
          (hardening_parameter +
           ((self_hardening) ? (1.0 - hardening_parameter) : 0.0)));
}



template <int dim>
inline double
ScalarMicroscopicStressLaw<dim>::sgn(
  const double value) const
{
  return (0.0 < value) - (value < 0.0);
}



template<int dim>
class VectorMicroscopicStressLaw
{
public:
  VectorMicroscopicStressLaw(
    const std::shared_ptr<CrystalsData<dim>>                      &crystals_data,
    const RunTimeParameters::VectorMicroscopicStressLawParameters parameters);

  void init();

  const dealii::SymmetricTensor<2,dim>
    &get_reduced_gradient_hardening_tensor(
      const unsigned int crystal_id,
      const unsigned int slip_id) const;

  const std::vector<dealii::SymmetricTensor<2,dim>>
    &get_reduced_gradient_hardening_tensors(
      const unsigned int crystal_id) const;

  dealii::Tensor<1,dim> get_vector_microscopic_stress(
    const unsigned int          crystal_id,
    const unsigned int          slip_id,
    const dealii::Tensor<1,dim> slip_gradient) const;

private:
  std::shared_ptr<const CrystalsData<dim>>    crystals_data;

  const double                                energetic_length_scale;

  const double                                initial_slip_resistance;

  std::vector<std::vector<dealii::SymmetricTensor<2,dim>>>
                                              reduced_gradient_hardening_tensors;

  bool                                        flag_init_was_called;
};



template <int dim>
inline const dealii::SymmetricTensor<2,dim>
&VectorMicroscopicStressLaw<dim>::
  get_reduced_gradient_hardening_tensor(
    const unsigned int crystal_id,
    const unsigned int slip_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The VectorMicroscopicStressLaw<dim> "
                                 "instance has not been initialized."));

  return (reduced_gradient_hardening_tensors[crystal_id][slip_id]);
}



template <int dim>
inline const std::vector<dealii::SymmetricTensor<2,dim>>
&VectorMicroscopicStressLaw<dim>::
  get_reduced_gradient_hardening_tensors(
    const unsigned int crystal_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The VectorMicroscopicStressLaw<dim> "
                                 "instance has not been initialized."));

  return (reduced_gradient_hardening_tensors[crystal_id]);
}



template<int dim>
class MicroscopicTractionLaw
{
public:
  MicroscopicTractionLaw(
    const std::shared_ptr<CrystalsData<dim>>                  &crystals_data,
    const RunTimeParameters::MicroscopicTractionLawParameters parameters);

  using GrainInteractionModuli =
    typename std::pair<std::vector<dealii::FullMatrix<double>>,
                       std::vector<dealii::FullMatrix<double>>>;

  GrainInteractionModuli get_grain_interaction_moduli(
    const unsigned int                  crystal_id_current_cell,
    const unsigned int                  crystal_id_neighbour_cell,
    std::vector<dealii::Tensor<1,dim>>  normal_vector_values) const;

  double get_microscopic_traction(
    const unsigned int                      q_point,
    const unsigned int                      slip_id_alpha,
    const GrainInteractionModuli            grain_interaction_moduli,
    const std::vector<std::vector<double>>  slip_values_current_cell,
    const std::vector<std::vector<double>>  slip_values_neighbour_cell) const;

  const dealii::FullMatrix<double> get_intra_gateaux_derivative(
    const unsigned int            q_point,
    const GrainInteractionModuli  grain_interaction_moduli) const;

  const dealii::FullMatrix<double> get_inter_gateaux_derivative(
    const unsigned int            q_point,
    const GrainInteractionModuli  grain_interaction_moduli) const;

private:
  std::shared_ptr<const CrystalsData<dim>>    crystals_data;

  const double                                grain_boundary_modulus;
};



template<int dim>
class CohesiveLaw
{
private:
  struct EffectiveQuantities
  {
    EffectiveQuantities(
      const double                          effective_opening_displacement,
      const dealii::Tensor<1,dim>           effective_direction,
      const dealii::SymmetricTensor<2,dim>  effective_identity_tensor,
      const double                          normal_opening_displacement)
    :
    opening_displacement(effective_opening_displacement),
    direction(effective_direction),
    identity_tensor(effective_identity_tensor),
    normal_opening_displacement(normal_opening_displacement)
    {}

    const double                          opening_displacement;

    const dealii::Tensor<1,dim>           direction;

    const dealii::SymmetricTensor<2,dim>  identity_tensor;

    double                                normal_opening_displacement;
  };

public:
  CohesiveLaw(
    const RunTimeParameters::CohesiveLawParameters parameters);

  dealii::Tensor<1,dim> get_cohesive_traction(
    const dealii::Tensor<1,dim> opening_displacement,
    const dealii::Tensor<1,dim> normal_vector,
    const double                max_effective_opening_displacement,
    const double                old_effective_opening_displacement,
    const double                time_step_size) const;

  dealii::SymmetricTensor<2,dim> get_current_cell_gateaux_derivative(
    const dealii::Tensor<1,dim> opening_displacement,
    const dealii::Tensor<1,dim> normal_vector,
    const double                max_effective_opening_displacement,
    const double                old_effective_opening_displacement,
    const double                time_step_size) const;

  dealii::SymmetricTensor<2,dim> get_neighbor_cell_gateaux_derivative(
    const dealii::Tensor<1,dim> opening_displacement,
    const dealii::Tensor<1,dim> normal_vector,
    const double                max_effective_opening_displacement,
    const double                old_effective_opening_displacement,
    const double                time_step_size) const;

  double get_degradation_function_value(const double damage_variable) const;

  double get_effective_opening_displacement(
    const dealii::Tensor<1,dim> opening_displacement,
    const dealii::Tensor<1,dim> normal_vector) const;

  EffectiveQuantities
    get_effective_quantities(
      const dealii::Tensor<1,dim> opening_displacement,
      const dealii::Tensor<1,dim> normal_vector) const;

private:

  double critical_cohesive_traction;

  double critical_opening_displacement;

  double tangential_to_normal_stiffness_ratio;

  double degradation_exponent;

  double macaulay_brackets(const double value) const;

  double get_effective_cohesive_traction(
    const double effective_opening_displacement) const;

  dealii::Tensor<1,dim> get_cohesive_traction_direction(
    const dealii::Tensor<1,dim> current_cell_displacement,
    const dealii::Tensor<1,dim> neighbor_cell_displacement,
    const dealii::Tensor<1,dim> normal_vector) const;
};



template <int dim>
inline double
CohesiveLaw<dim>::get_degradation_function_value(
  const double damage_variable) const
{
  return std::pow(1.0 - damage_variable, degradation_exponent);
}



template <int dim>
inline double
CohesiveLaw<dim>::macaulay_brackets(const double value) const
{
  if (value > 0)
    return value;
  else
    return 0.0;
}



template <int dim>
inline double
CohesiveLaw<dim>::get_effective_cohesive_traction(
  const double effective_opening_displacement) const
{
  return (critical_cohesive_traction *
          effective_opening_displacement /
          critical_opening_displacement *
          std::exp(1.0 - effective_opening_displacement /
                         critical_opening_displacement));
}



} // ConstitutiveLaws



} // gCP



#endif /* INCLUDE_CONSTITUTIVE_EQUATIONS_H_ */