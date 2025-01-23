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
  ElasticStrain(std::shared_ptr<CrystalsData<dim>>  crystals_data,
  const double dimensionless_number = 1.0);

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

  double                                      dimensionless_number;
};



} // namespace Kinematics


namespace ConstitutiveLaws
{



template<int dim>
class HookeLaw
{
public:
  HookeLaw(const RunTimeParameters::HookeLawParameters  parameters,
           const double characteristic_stiffness = 1.0);

  HookeLaw(const std::shared_ptr<CrystalsData<dim>>     &crystals_data,
           const RunTimeParameters::HookeLawParameters  parameters,
           const double characteristic_stiffness = 1.0);

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

  double                                      characteristic_stiffness;

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
class ScalarMicrostressLaw
{
public:
  ScalarMicrostressLaw(
    const std::shared_ptr<CrystalsData<dim>>                &crystals_data,
    const RunTimeParameters::ScalarMicrostressLawParameters parameters,
    const RunTimeParameters::HardeningLaw                   hardening_law_prm,
    const double characteristic_slip_resistance = 1.0);

  double get_scalar_microstress(
    const double slip_value,
    const double old_slip_value,
    const double slip_resistance,
    const double time_step_size);

  dealii::FullMatrix<double> get_jacobian(
    const unsigned int                      q_point,
    const std::vector<std::vector<double>>  slip_values,
    const std::vector<std::vector<double>>  old_slip_values,
    const std::vector<double>               slip_resistances,
    const double                            time_step_size);

private:
  std::shared_ptr<const CrystalsData<dim>>  crystals_data;

  RunTimeParameters::RegularizationFunction regularization_function;

  const double                              regularization_parameter;

  const double                              linear_hardening_modulus;

  const double                              hardening_parameter;

  const double characteristic_slip_resistance;

  const bool                                flag_perfect_plasticity;

  const bool                                flag_rate_independent;

  double get_hardening_matrix_entry(const bool self_hardening) const;

  double sgn(const double value) const;

  double get_regularization_function_value(const double slip_rate) const;

  double get_regularization_function_derivative_value(const double slip_rate) const;
};



template <int dim>
inline double
ScalarMicrostressLaw<dim>::get_hardening_matrix_entry(
  const bool self_hardening) const
{
  return (linear_hardening_modulus /
          characteristic_slip_resistance *
          (hardening_parameter +
           ((self_hardening) ? (1.0 - hardening_parameter) : 0.0)));
}



template <int dim>
inline double
ScalarMicrostressLaw<dim>::sgn(
  const double value) const
{
  return (0.0 < value) - (value < 0.0);
}



template<int dim>
class VectorialMicrostressLaw
{
public:
  VectorialMicrostressLaw(
    const std::shared_ptr<CrystalsData<dim>> &crystals_data,
    const RunTimeParameters::VectorialMicrostressLawParameters parameters);

  void init(const bool flag_dimensionless_formulation = false);

  dealii::Tensor<1,dim> get_vectorial_microstress(
    const unsigned int          crystal_id,
    const unsigned int          slip_id,
    const dealii::Tensor<1,dim> slip_gradient) const;

  dealii::SymmetricTensor<2,dim> get_jacobian(
    const unsigned int          crystal_id,
    const unsigned int          slip_id,
    const dealii::Tensor<1,dim> slip_gradient) const;

  std::vector<dealii::SymmetricTensor<2,dim>> get_jacobian(
    const unsigned int          crystal_id,
    const unsigned int          slip_id,
    const std::vector<dealii::Tensor<1,dim>> slip_gradient) const;

private:
  std::shared_ptr<const CrystalsData<dim>>    crystals_data;

  const double                                energetic_length_scale;

  const double                                initial_slip_resistance;

  const double                                defect_energy_index;

  double                                      factor;

  std::vector<std::vector<dealii::SymmetricTensor<2,dim>>>
                                              slip_direction_dyads;

  std::vector<std::vector<dealii::SymmetricTensor<2,dim>>>
                                              slip_binormal_dyads;

  bool                                        flag_init_was_called;
};



template<int dim>
class MicrotractionLaw
{
public:
  MicrotractionLaw(
    const std::shared_ptr<CrystalsData<dim>> &crystals_data,
    const RunTimeParameters::MicrotractionLawParameters &parameters,
    const double characteristic_vectorial_microstress = 1.0);

  using GrainInteractionModuli =
    typename std::pair<std::vector<dealii::FullMatrix<double>>,
                       std::vector<dealii::FullMatrix<double>>>;

  GrainInteractionModuli get_grain_interaction_moduli(
    const unsigned int                  crystal_id_current_cell,
    const unsigned int                  crystal_id_neighbour_cell,
    std::vector<dealii::Tensor<1,dim>>  normal_vector_values) const;

  double get_microtraction(
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

  double get_free_energy_density(
    const unsigned int                      neighbor_cell_crystal_id,
    const unsigned int                      current_cell_crystal_id,
    const unsigned int                      quadrature_point_id,
    std::vector<dealii::Tensor<1,dim>>      normal_vector_values,
    const std::vector<std::vector<double>>  neighbor_cell_slip_values,
    const std::vector<std::vector<double>>  current_cell_slip_values) const;

private:
  std::shared_ptr<const CrystalsData<dim>> crystals_data;

  const double grain_boundary_modulus;

  const double characteristic_vectorial_microstress;
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
    const RunTimeParameters::CohesiveLawParameters parameters,
    const double characteristic_stress = 1.0,
    const double characteristic_displacement = 1.0);

  dealii::Tensor<1,dim> get_cohesive_traction(
    const dealii::Tensor<1,dim> opening_displacement,
    const dealii::Tensor<1,dim> normal_vector,
    const double                max_effective_opening_displacement,
    const double                old_effective_opening_displacement,
    const double                time_step_size) const;

  dealii::SymmetricTensor<2,dim> get_jacobian(
    const dealii::Tensor<1,dim> opening_displacement,
    const dealii::Tensor<1,dim> normal_vector,
    const double                max_effective_opening_displacement,
    const double                old_effective_opening_displacement,
    const double                time_step_size) const;

  double get_free_energy_density(
    const double effective_opening_displacement) const;

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

  double characteristic_stress;

  double characteristic_displacement;

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
          std::exp(1.0 -
            effective_opening_displacement /
            critical_opening_displacement));
}



class DegradationFunction
{
public:

  DegradationFunction(
    const RunTimeParameters::DegradationFunction parameters);

  double get_degradation_function_value(
    const double  damage_variable,
    const bool    flag_couple_damage) const;

  double get_degradation_function_derivative_value(
    const double  damage_variable,
    const bool    flag_couple_damage) const;

private:

  const double degradation_exponent;
};



inline double
DegradationFunction::get_degradation_function_value(
  const double  damage_variable,
  const bool    flag_couple_damage) const
{
  if (flag_couple_damage)
  {
    return std::pow(1.0 - damage_variable, degradation_exponent);
  }
  else
  {
    return 1.0;
  }
}



inline double
DegradationFunction::get_degradation_function_derivative_value(
  const double  damage_variable,
  const bool    flag_couple_damage) const
{
  if (flag_couple_damage)
  {
    return (- degradation_exponent *
           std::pow(1.0 - damage_variable, degradation_exponent - 1.0));
  }
  else
  {
    return 1.0;
  }
}



/*!
 * @brief The constitutive law describing the traction product of the
 * mechanical contact of two bodies or one body with itself.
 *
 * @details The normal contact between two bodies is numerically
 * considered through the penalty method by introducing the free energy
 * density
 *  \f[
 *      \psi_{\mathrm{C}} =
 *      \int_{\mathrm{I}} \frac{1}{2} \varepsilon k_0
 *      \left\langle - \delta_{\mathrm{n}} \right\rangle^2 \,\mathrm{d}a
 *  \f]
 * at the contact surfaces, where \f$ \varepsilon \f$ is the penalty
 * coefficient, \f$ k_0 \f$ the reference stiffness and
 * \f$ \delta_{\mathrm{n}} \f$ the normal component of the displacement
 * jump between bodies.
 * @note Currently only frictionless contact implemented and
 * node-to-node discretization at the grain boundaries
 *
 * @todo Frictional contact
 *
 * @tparam dim Spatial dimension
 */
template<int dim>
class ContactLaw
{
public:
  /*!
   * @brief Constructor
   *
   * @param parameters The constitutive law's parameters
   */
  ContactLaw(
    const RunTimeParameters::ContactLawParameters parameters,
    const double characteristic_stress = 1.0,
    const double characteristic_displacement = 1.0);

  /*!
   * @brief Method returning the contact traction
   *
   * @details It is computed as
   *  \f[
   *      \bs{t}_{\mathrm{c}} = - \varepsilon k_0
   *      \macaulay{-\delta_{\mathrm{n}}} \bs{n}
   *  \f]
   *
   * @param opening_displacement Opening displacement at the evaluation
   * point \f$ \bs{\delta} \f$
   * @param normal_vector Normal vector at the evaluation point
   * \f$ \bs{n} \f$
   * @return dealii::Tensor<1,dim> Contact traction at the evaluation
   * point \f$ \bs{t}_{\mathrm{c}} \f$
   */
  dealii::Tensor<1,dim> get_contact_traction(
    const dealii::Tensor<1,dim> opening_displacement,
    const dealii::Tensor<1,dim> normal_vector) const;

  /*!
   * @brief Method returning the Gateaux derivative of the
   * contact traction with respect to the current cell
   *
   * @details It is computed as
   *  \f[
   *      \bs{J}_{\bs{t}_\mathrm{c}} =  \varepsilon
   *      \macaulay{-\frac{\delta_{\mathrm{n}}}{\abs{\delta_{\mathrm{n}}}}}
   *      \bs{n} \otimes \bs{n}
   *  \f]
   *
   * @param opening_displacement Opening displacement at the evaluation
   * point \f$ \bs{\delta} \f$
   * @param normal_vector Normal vector at the evaluation point
   * \f$ \bs{n} \f$
   * @return dealii::SymmetricTensor<2,dim>  Contact traction at the evaluation
   * point \f$ \bs{J}_{\bs{t}_\mathrm{c}} \f$
   */
  dealii::SymmetricTensor<2,dim> get_jacobian(
    const dealii::Tensor<1,dim> opening_displacement,
    const dealii::Tensor<1,dim> normal_vector) const;

private:
  /*!
   * @brief The penalty coefficient multiplying the @ref stiffness value
   * leading to the effective stiffness
   */
  double penalty_coefficient;

  /**
   * @brief
   *
   */
  double characteristic_stress;

  /**
   * @brief
   *
   */
  double characteristic_displacement;

  /*!
  * @brief A method returning the result of applying the Macaulay
  * brackets to the input variable
  *
  * @details They are defined as
  *
  *  \f[
  *      \left\langle a \right\rangle =
  *      \begin{cases}
  *        a, & a > 0 \\
  *        0, & a < 0
  *      \end{cases}
  *  \f]
  *
  * @param value Input value
  * @return double Output value
  */
  double macaulay_brackets(const double value) const;

};



template <int dim>
inline double
ContactLaw<dim>::macaulay_brackets(const double value) const
{
  if (value > 0.)
    return value;
  else
    return 0.0;
}



} // ConstitutiveLaws



} // gCP



#endif /* INCLUDE_CONSTITUTIVE_EQUATIONS_H_ */