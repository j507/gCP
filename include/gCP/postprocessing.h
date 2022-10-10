#ifndef INCLUDE_POSTPROCESSING_H_
#define INCLUDE_POSTPROCESSING_H_

#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_postprocessor.h>

namespace gCP
{



namespace Postprocessing
{



template <int dim>
class Postprocessor : public dealii::DataPostprocessor<dim>
{
public:
  Postprocessor(
    std::shared_ptr<FEField<dim>>       &fe_field,
    std::shared_ptr<CrystalsData<dim>>  &crystals_data);

  virtual void evaluate_vector_field(
    const dealii::DataPostprocessorInputs::Vector<dim>  &inputs,
    std::vector<dealii::Vector<double>>                 &computed_quantities)
    const override;

  virtual std::vector<std::string> get_names() const override;

  virtual std::vector<
    dealii::DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const override;

  virtual dealii::UpdateFlags get_needed_update_flags() const override;

  void init(
    std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law);

private:
  std::shared_ptr<const FEField<dim>>                     fe_field;

  std::shared_ptr<const CrystalsData<dim>>                crystals_data;

  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law;

  std::vector<std::pair<unsigned int, unsigned int>>      voigt_indices;

  const dealii::SymmetricTensor<4,dim>                    deviatoric_projector;

  const dealii::SymmetricTensor<4,3>                      deviatoric_projector_3d;

  bool                                                    flag_init_was_called;

  dealii::SymmetricTensor<2,3> convert_2d_to_3d(
    dealii::SymmetricTensor<2,dim> symmetric_tensor) const;

  double get_von_mises_stress(
    const dealii::SymmetricTensor<2,3> stress_tensor_in_3d) const;

  double get_von_mises_plastic_strain(
    const dealii::SymmetricTensor<2,dim> strain_tensor) const;
};



template <int dim>
class SimpleShear
{
public:
  SimpleShear(
    std::shared_ptr<FEField<dim>>         &fe_field,
    std::shared_ptr<dealii::Mapping<dim>> &mapping,
    const double                          max_shear_strain_at_upper_boundary,
    const double                          min_shear_strain_at_upper_boundary,
    const double                          period,
    const double                          initial_loading_time,
    const RunTimeParameters::LoadingType  loading_type,
    const dealii::types::boundary_id      upper_boundary_id,
    const double                          width);

  void init(
    std::shared_ptr<const Kinematics::ElasticStrain<dim>>   elastic_strain,
    std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law);

  void compute_data(const double time);

  void output_data_to_file(std::ostream &file) const;

private:
  dealii::TableHandler                                    table_handler;

  std::shared_ptr<const FEField<dim>>                     fe_field;

  const dealii::hp::MappingCollection<dim>                mapping_collection;

  std::shared_ptr<const Kinematics::ElasticStrain<dim>>   elastic_strain;

  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law;

  const double                                            max_shear_strain_at_upper_boundary;

  const double                                            min_shear_strain_at_upper_boundary;

  const double                                            period;

  const double                                            initial_loading_time;

  RunTimeParameters::LoadingType                          loading_type;

  const dealii::types::boundary_id                        upper_boundary_id;

  double                                                  average_stress_12;

  const double                                            width;

  bool                                                    flag_init_was_called;

  void compute_stress_12_at_boundary();
};



}  // Postprocessing



} // gCP



#endif /* INCLUDE_POSTPROCESSING_H_ */