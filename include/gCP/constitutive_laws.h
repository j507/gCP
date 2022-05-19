#ifndef INCLUDE_CONSTITUTIVE_EQUATIONS_H_
#define INCLUDE_CONSTITUTIVE_EQUATIONS_H_

#include <gCP/crystal_data.h>
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
  using ExtractorPair =
    typename std::pair<
      const std::vector<dealii::FEValuesExtractors::Vector>,
      const std::vector<std::vector<dealii::FEValuesExtractors::Scalar>>>;

  ElasticStrain(std::shared_ptr<CrystalsData<dim>>  crystals_data);

  void init(ExtractorPair &extractor_pair);

  const std::vector<dealii::SymmetricTensor<2,dim>> get_elastic_strain_tensor(
    const dealii::LinearAlgebraTrilinos::MPI::Vector  solution,
    const dealii::FEValues<dim>                       &fe_values,
    const dealii::types::material_id                  crystal_id) const;

private:

  std::shared_ptr<const CrystalsData<dim>>    crystals_data;

  std::vector<dealii::FEValuesExtractors::Vector>
                                              displacements_extractors;

  std::vector<std::vector<dealii::FEValuesExtractors::Scalar>>
                                              slips_extractors;

  bool                                        flag_init_was_called;
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

  return (stress_tensor *
          crystals_data->get_symmetrized_schmid_tensor(crystal_id,
                                                       slip_id));
}



template<int dim>
class ScalarMicroscopicStressLaw
{
public:
  ScalarMicroscopicStressLaw(
    const std::shared_ptr<CrystalsData<dim>>  &crystals_data,
    const std::string                         regularization_function,
    const double                              regularization_parameter,
    const double                              initial_slip_resistance,
    const double                              linear_hardening_modulus,
    const double                              hardening_parameter);

  void init();

  double get_scalar_microscopic_stress(const double crystal_id,
                                       const double slip_id,
                                       const double slip_rate);

  double get_gateaux_derivative(const unsigned int  crystald_id,
                                const unsigned int  slip_id,
                                const bool          self_hardening,
                                const double        slip_rate_alpha,
                                const double        slip_rate_beta,
                                const double        time_step_size);

private:
  enum class RegularizationFunction
  {
    PowerLaw,
    Tanh,
  };

  std::shared_ptr<const CrystalsData<dim>>  crystals_data;

  RegularizationFunction                    regularization_function;

  const double                              regularization_parameter;

  const double                              initial_slip_resistance;

  const double                              linear_hardening_modulus;

  const double                              hardening_parameter;

  std::vector<std::vector<double>>          hardening_field_at_q_points;

  bool                                      flag_init_was_called;

  double get_hardening_matrix_entry(const bool self_hardening) const;
};



template <int dim>
inline double
ScalarMicroscopicStressLaw<dim>::get_hardening_matrix_entry(
  const bool self_hardening) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The ScalarMicroscopicStressLaw<dim> "
                                 "instance has not been initialized."));

  return (linear_hardening_modulus *
          (hardening_parameter +
           (self_hardening) ? (1.0 - hardening_parameter) : 0.0));
}




template<int dim>
class VectorMicroscopicStressLaw
{
public:
  VectorMicroscopicStressLaw(
    const std::shared_ptr<CrystalsData<dim>>  &crystals_data,
    const double                              energetic_length_scale,
    const double                              initial_slip_resistance);

  void init();

  const dealii::SymmetricTensor<2,dim>
    &get_reduced_gradient_hardening_tensor(
      const unsigned int crystal_id,
      const unsigned int slip_id) const;

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



template<int dim>
class MicroscopicTractionLaw
{
public:
  MicroscopicTractionLaw();
};



template<int dim>
class MicroscopicInterfaceTractionLaw
{
public:
  MicroscopicInterfaceTractionLaw();
};



} // ConstitutiveLaws



} // gCP



#endif /* INCLUDE_CONSTITUTIVE_EQUATIONS_H_ */