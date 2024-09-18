#include <gCP/boundary_conditions.h>



namespace gCP
{



namespace BoundaryConditions
{



template<int dim>
DisplacementControl<dim>::DisplacementControl(
  const RunTimeParameters::SimpleLoading  &parameters,
  const unsigned int                      n_components,
  const unsigned int                      n_crystals,
  const double                            strip_height,
  const bool                              flag_is_decohesion_allowed,
  const double                            characteristic_displacement,
  const unsigned int                      component)
:
dealii::Function<dim>(n_components, parameters.start_time),
n_crystals(n_crystals),
strip_height(strip_height),
loading_type(RunTimeParameters::LoadingType::Monotonic),
max_strain_load(parameters.max_load),
min_strain_load(parameters.min_load),
component(component),
period(parameters.period),
start_of_cyclic_phase(parameters.start_of_cyclic_phase),
start_of_unloading(parameters.start_of_unloading_phase),
characteristic_displacement(characteristic_displacement),
flag_is_decohesion_allowed(flag_is_decohesion_allowed)
{
  Assert(
  component < dim,
  dealii::ExcMessage("The component can't be higher than the "
                      "dimension of the Euclidean space"));
}



template<int dim>
void DisplacementControl<dim>::vector_value(
  const dealii::Point<dim>  &/*point*/,
  dealii::Vector<double>    &return_vector) const
{
  const double time = this->get_time();

  return_vector = 0.0;

  double displacement_load = 0.0;

  switch (loading_type)
  {
    case RunTimeParameters::LoadingType::Monotonic:
      {
        displacement_load =
          time * strip_height * max_strain_load;
      }
      break;
    case RunTimeParameters::LoadingType::Cyclic:
      {
        const double peak_amplitude =
          (max_strain_load - min_strain_load) / 2.0;

        const double offset =
          (max_strain_load + min_strain_load) / 2.0;

        if (time < start_of_cyclic_phase)
        {
          displacement_load = time * strip_height * offset;
        }
        else if (time < start_of_unloading)
        {
          displacement_load = peak_amplitude *
                              std::sin(2.0 * M_PI / period *
                                      (time - start_of_cyclic_phase))
                              +
                              offset;
        }
        else
        {
          displacement_load = offset * (1 - time + start_of_unloading);
        }
      }
      break;
    default:
      Assert(false, dealii::ExcNotImplemented());
  }

  return_vector[component] = displacement_load /
    characteristic_displacement;

  if (flag_is_decohesion_allowed)
  {
    for (unsigned int i = 1; i < n_crystals; ++i)
    {
      return_vector[component + i*dim] =
        displacement_load / characteristic_displacement;
    }
  }
}



template<int dim>
LoadControl<dim>::LoadControl(
  const RunTimeParameters::SimpleLoading &parameters,
  const bool flag_is_decohesion_allowed,
  const double characteristic_traction)
:
dealii::TensorFunction<1, dim>(parameters.start_time),
loading_type(RunTimeParameters::LoadingType::Monotonic),
max_traction_load(0.),
min_traction_load(0.),
component(0.),
duration_of_loading_and_unloading_phase(0.),
period(0.),
start_of_cyclic_phase(0.),
start_of_unloading(0.),
characteristic_traction(characteristic_traction),
flag_is_decohesion_allowed(flag_is_decohesion_allowed)
{
  Assert(
    component < dim,
    dealii::ExcMessage("The component can't be higher than the "
                       "dimension of the Euclidean space"));
}



template<int dim>
dealii::Tensor<1, dim> LoadControl<dim>::value(
  const dealii::Point<dim>  &/*point*/) const
{
  const double time = this->get_time();

  dealii::Tensor<1, dim> return_vector;

  double traction_load = 0.0;

  switch (loading_type)
  {
    case RunTimeParameters::LoadingType::Monotonic:
      {
        traction_load =
          time * max_traction_load;
      }
      break;
    case RunTimeParameters::LoadingType::Cyclic:
      {
        const double peak_amplitude =
          (max_traction_load - min_traction_load) / 2.0;

        const double offset =
          (max_traction_load + min_traction_load) / 2.0;

        if (time < start_of_cyclic_phase)
        {
          traction_load = time * offset;
        }
        else if (time < start_of_unloading)
        {
          traction_load = peak_amplitude *
                              std::sin(2.0 * M_PI / period *
                                      (time - start_of_cyclic_phase))
                              +
                              offset;
        }
        else
        {
          traction_load = offset * (1 - time + start_of_unloading);
        }
      }
      break;
    default:
      Assert(false, dealii::ExcNotImplemented());
  }

  return_vector[component] = traction_load / characteristic_traction;

  return return_vector;
}



} // namespace BoundaryConditions



} // namespace gCP



template class gCP::BoundaryConditions::DisplacementControl<2>;
template class gCP::BoundaryConditions::DisplacementControl<3>;

template class gCP::BoundaryConditions::LoadControl<2>;
template class gCP::BoundaryConditions::LoadControl<3>;