#ifndef INCLUDE_POSTPROCESSING_H_
#define INCLUDE_POSTPROCESSING_H_

#include <gCP/constitutive_laws.h>
#include <gCP/fe_field.h>

#include <deal.II/base/discrete_time.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

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
    std::shared_ptr<FEField<dim>> &fe_field,
    std::shared_ptr<CrystalsData<dim>> &crystals_data,
    const RunTimeParameters::DimensionlessForm &parameters,
    const bool flag_light_output = false,
    const bool flag_output_dimensionless_quantities = false,
    const bool flag_output_fluctuations = false);

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

  void set_macroscopic_strain(
    const dealii::SymmetricTensor<2,dim> macroscopic_strain);

private:
  std::shared_ptr<const FEField<dim>> fe_field;

  std::shared_ptr<const CrystalsData<dim>> crystals_data;

  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>> hooke_law;

  std::vector<std::pair<unsigned int, unsigned int>> voigt_indices;

  dealii::SymmetricTensor<2,dim> macroscopic_strain;

  const dealii::SymmetricTensor<4,dim> deviatoric_projector;

  const dealii::SymmetricTensor<4,3> deviatoric_projector_3d;

  RunTimeParameters::DimensionlessForm parameters;

  bool flag_light_output;

  bool flag_output_dimensionless_quantities;

  bool flag_output_fluctuations;

  bool flag_init_was_called;

  dealii::SymmetricTensor<2,3> convert_2d_to_3d(
    dealii::SymmetricTensor<2,dim> symmetric_tensor) const;

  double get_von_mises_stress(
    const dealii::SymmetricTensor<2,3> stress_tensor_in_3d) const;

  double get_von_mises_plastic_strain(
    const dealii::SymmetricTensor<2,dim> strain_tensor) const;
};



template <int dim>
class SlipBasedPostprocessor :  public dealii::DataPostprocessor<dim>
{
public:
  void reinit(
    std::shared_ptr<const CrystalsData<dim>> &crystals_data,
    const std::string output_name,
    const unsigned int n_components,
    const bool flag_allow_decohesion);

  virtual void evaluate_vector_field(
    const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<dealii::Vector<double>> &computed_quantities)
    const override;

  virtual std::vector<std::string> get_names() const override;

  virtual std::vector<
    dealii::DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const override;

  virtual dealii::UpdateFlags get_needed_update_flags() const override;

private:
  std::shared_ptr<const CrystalsData<dim>> crystals_data;

  std::string output_name;

  unsigned int n_components;

  bool flag_allow_decohesion;
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


/*!
 * @brief A class for the micro-to-macro homogenization
 *
 * @tparam dim Spatial dimension
 *
 * @note Only strain driven homoganization is considered in this class
 *
 * @todo Methods to compute macroscopic jacobian
 */
template <int dim>
class Homogenization
{
public:
  /*!
   * @brief Constructor
   *
   * @param fe_field Shared pointer to the @ref FEField instance
   * @param mapping Shared pointer to the @ref dealii::Mapping instance
   * @todo Docu
   */
  Homogenization(
    std::shared_ptr<FEField<dim>>         &fe_field,
    std::shared_ptr<dealii::Mapping<dim>> &mapping);

  /*!
   * @brief Method initiatating the class instance for further use
   *
   * @param elastic_strain Shared pointer to the
   * @ref Kinematics::ElasticStrain instance
   * @param hooke_law Shared pointer to the
   * @ref ConstitutiveLaws::HookeLaw instance
   * @param path_to_output_file Output file stream to the file's path
   * in which data will be written to
   * @details Sets the internal shared pointers to the method's arguments
   */
  void init(
    std::shared_ptr<const Kinematics::ElasticStrain<dim>>   elastic_strain,
    std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law,
    std::ofstream                                           &path_to_output_file);

  /*!
   * @brief Method computing the macroscopic stress and stiffness tetrad
   */
  void compute_macroscopic_quantities(const double time);

  /*!
   * @brief Prints the macroscopic quantities (stored in
   * @ref table_handler) to the file specified by
   * @ref path_to_output_file
   *
   * @param time Time to which the macroscopic quantities are assigned to
   */
  void output_macroscopic_quantities_to_file();

  /*!
   * @brief Sets the macroscopic strain
   *
   * @param macroscopic_strain Macroscopic strain
   */
  void set_macroscopic_strain(
    const dealii::SymmetricTensor<2,dim> macroscopic_strain);

  /*!
   * @brief Getter returning @ref macroscopic_stress
   *
   * @return const dealii::SymmetricTensor<2,dim>& The current value of
   * the macroscopic stress
   */
  const dealii::SymmetricTensor<2,dim> &get_macroscopic_stress() const;

  /*!
   * @brief Getter returning @ref macroscopic_stiffness_tetrad
   *
   * @return const dealii::SymmetricTensor<4,dim>& The current value of
   * the macroscopic stiffness tetrad
   */
  const dealii::SymmetricTensor<4,dim> &get_macroscopic_stiffness_tetrad() const;

private:
  /*!
  * @brief Shared pointer to @ref FEField instance
  */
  std::shared_ptr<const FEField<dim>>                     fe_field;

  /*!
   * @brief The collection of mappings specifying how the reference
   * cells are imbuded into the tessellation.
   */
  const dealii::hp::MappingCollection<dim>                mapping_collection;

  /*!
   * @brief Collection of quadrature formulas required for the numerical
   * integration
   */
  dealii::hp::QCollection<dim>                            quadrature_collection;

  /*!
   * @brief Shared pointer to the elastic strain measure,
   * i.e., @ref Kinematics::ElasticStrain
   */
  std::shared_ptr<const Kinematics::ElasticStrain<dim>>   elastic_strain;

  /*!
   * @brief Shared pointer to the constitutive law for the stress tensor,
   * i.e., @ref ConstitutiveLaws::HookeLaw
   */
  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law;

  /*!
   * @brief The macroscopic stress tensor
   * @details See @ref compute_macroscopic_stress for its definition
   */
  dealii::SymmetricTensor<2,dim>                          macroscopic_stress;

  /*!
   * @brief The macroscopic stress tensor
   * @details See @ref compute_macroscopic_stress for its definition
   */
  dealii::SymmetricTensor<2,dim>                          microstress_fluctuations;

  /*!
   * @brief The macroscopic strain tensor
   */
  dealii::SymmetricTensor<2,dim>                          macroscopic_strain;

  /*!
   * @brief The microscopic strain fluctuations
   */
  dealii::SymmetricTensor<2,dim>                          microstrain_fluctuations;

  /*!
   * @brief The macroscopic stiffness tetrad
   * @details See @ref compute_macroscopic_stiffness_tetrad for its definition
   */
  dealii::SymmetricTensor<4,dim>                          macroscopic_stiffness_tetrad;

  /*!
   * @brief The deviatoric projector
   *
   * @details Defined as
   * \f[
   * \ts{P}{4}_{\mathrm{dev}} = \ts{1}{4} - \frac{1}{3} \bs{1} \otimes \bs{1}
   * \f]
   *    * @note It is used to compute the Von-Mises stress and strain
   */
  dealii::SymmetricTensor<4,dim>                          deviatoric_projector;

  /*!
   * @brief The @ref dealii::TableHandler instance for data storage and
   * output
   */
  dealii::TableHandler                                    table_handler;

  /*!
   * @brief Path to the file where the data of @ref table_handler
   * is written
   */
  std::ofstream                                           path_to_output_file;

  /*!
   * @brief Boolean indicating if the class was initialized
   */
  bool                                                    flag_init_was_called;

  /*!
   * @brief Method to compute the macroscopic stress
   *
   * @details Computed as
  * \f[
  * \overbar{\bs{T}} =
  * \frac{1}{\vol (\mathcal{B})}\int_{\mathcal{B}} \bs{T} \d v
  * \f]
  */
  void compute_macroscopic_stress();

  /*!
   * @brief Method to compute the macroscopic stiffness tetrad
   *
   * @todo Implement an homogenization scheme. Method is currently empty
   */
  void compute_macroscopic_stiffness_tetrad();

  /*!
   * @brief Updates the @ref table_handler with the macroscopic
   * quantities
   *
   * @details The components of the strain and strain tensor as well as
   * their Von-Mises equivalents are added to the @ref table_handler
   *
   * @param time Time associated with the updated values
   */
  void update_table_handler_values(const double time);
};



template <int dim>
inline const dealii::SymmetricTensor<2,dim>
&Homogenization<dim>::get_macroscopic_stress() const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The HookeLaw<dim> instance has not"
                                 " been initialized."));

  return (macroscopic_stress);
}



template <int dim>
inline const dealii::SymmetricTensor<4,dim>
&Homogenization<dim>::get_macroscopic_stiffness_tetrad() const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The HookeLaw<dim> instance has not"
                                 " been initialized."));

  return (macroscopic_stiffness_tetrad);
}



}  // Postprocessing



} // gCP



#endif /* INCLUDE_POSTPROCESSING_H_ */
