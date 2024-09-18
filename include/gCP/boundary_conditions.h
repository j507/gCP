#ifndef INCLUDE_BOUNDARY_CONDITIONS_H_
#define INCLUDE_BOUNDARY_CONDITIONS_H_



#include <gCP/run_time_parameters.h>



#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>



namespace gCP
{



namespace BoundaryConditions
{



template <int dim>
class DisplacementControl : public dealii::Function<dim>
{
public:
  DisplacementControl(
    const RunTimeParameters::SimpleLoading &parameters,
    const unsigned int n_components,
    const unsigned int n_crystals,
    const double strip_height,
    const bool flag_is_decohesion_allowed,
    const double characteristic_displacement = 1.0,
    const unsigned int component = 0);

  virtual void vector_value(
    const dealii::Point<dim>  &point,
    dealii::Vector<double>    &return_vector) const override;

private:
  const unsigned int                    n_crystals;

  const double                          strip_height;

  const RunTimeParameters::LoadingType  loading_type;

  const double                          max_strain_load;

  const double                          min_strain_load;

  const unsigned int                    component;

  const double                          period;

  const double                          start_of_cyclic_phase;

  const double                          start_of_unloading;

  const double                          characteristic_displacement;

  const bool                            flag_is_decohesion_allowed;
};




template <int dim>
class LoadControl : public dealii::TensorFunction<1,dim>
{
public:
  LoadControl(
    const RunTimeParameters::SimpleLoading &parameters,
    const bool flag_is_decohesion_allowed,
    const double characteristic_traction = 1.0);

  virtual dealii::Tensor<1, dim> value(
    const dealii::Point<dim>  &point) const override;

private:

  const RunTimeParameters::LoadingType  loading_type;

  const double                          max_traction_load;

  const double                          min_traction_load;

  const int                             component;

  const double                          duration_of_loading_and_unloading_phase;

  const double                          period;

  const double                          start_of_cyclic_phase;

  const double                          start_of_unloading;

  const double                          characteristic_traction;

  const bool                            flag_is_decohesion_allowed;
};



} // namespace BoundaryConditions



} // namespace gCP



#endif /* INCLUDE_BOUNDARY_CONDITIONS_H_ */
