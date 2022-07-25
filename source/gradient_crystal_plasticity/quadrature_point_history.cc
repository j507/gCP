
#include <gCP/quadrature_point_history.h>

#include <deal.II/grid/filtered_iterator.h>

namespace gCP
{



template <int dim>
void InterfaceData<dim>::init(const double value)
{
  this->value = value;
}



template <int dim>
void InterfaceData<dim>::prepare_for_update_call()
{
  was_updated = false;
}



template <int dim>
void InterfaceData<dim>::update(
  const dealii::Tensor<1,dim> a,
  const dealii::Tensor<1,dim> b)
{
  if (was_updated)
    return;

  const dealii::Tensor<1,dim> c = a-b;

  value = std::max(value, c.norm());

  was_updated = true;
}



template <int dim>
InterfaceQuadraturePointHistory<dim>::InterfaceQuadraturePointHistory()
:
max_effective_opening_displacement(0.0),
damage_variable(0.0),
flag_init_was_called(false)
{}



template <int dim>
void InterfaceQuadraturePointHistory<dim>::init(
  const RunTimeParameters::DecohesionLawParameters &parameters)
{
  if (flag_init_was_called)
    return;

  critical_cohesive_traction    = parameters.critical_cohesive_traction;

  critical_opening_displacement = parameters.critical_opening_displacement;

  /*critical_energy_release_rate  = critical_cohesive_traction *
                                  critical_opening_displacement *
                                  std::exp(1.0);*/

  flag_init_was_called          = true;
}



template <int dim>
void InterfaceQuadraturePointHistory<dim>::store_current_values()
{
  tmp_values = std::make_pair(max_effective_opening_displacement,
                              damage_variable);

  flag_values_were_updated = false;
}



template <int dim>
void InterfaceQuadraturePointHistory<dim>::update_values(
  const dealii::Tensor<1,dim> neighbor_cell_displacement,
  const dealii::Tensor<1,dim> current_cell_displacement)
{
  /*if (flag_values_were_updated)
    return;*/

  max_effective_opening_displacement  = tmp_values.first;
  damage_variable                     = tmp_values.second;

  const dealii::Tensor<1,dim> displacement_jump =
    neighbor_cell_displacement - current_cell_displacement;

  max_effective_opening_displacement =
    std::max(max_effective_opening_displacement,
             displacement_jump.norm());

  const double displacement_ratio =
     max_effective_opening_displacement / critical_opening_displacement;

  /*
  const double free_energy_density =
    std::exp(1.0) *
    critical_cohesive_traction *
    critical_opening_displacement *
    (1.0 - (1.0 + displacement_ratio) * std::exp(-displacement_ratio));

  damage_variable = free_energy_density/critical_energy_release_rate;
  */

  damage_variable =
    1.0 - (1.0 + displacement_ratio) * std::exp(-displacement_ratio);

  flag_values_were_updated = true;
}



template <int dim>
QuadraturePointHistory<dim>::QuadraturePointHistory()
:
flag_init_was_called(false)
{}



template <int dim>
void QuadraturePointHistory<dim>::init(
  const RunTimeParameters::ScalarMicroscopicStressLawParameters
    &parameters,
  const unsigned int n_slips)
{
  this->n_slips             = n_slips;

  initial_slip_resistance   = parameters.initial_slip_resistance;

  linear_hardening_modulus  = parameters.linear_hardening_modulus;

  hardening_parameter       = parameters.hardening_parameter;

  slip_resistances          = std::vector<double>(
                                n_slips,
                                0.0 /*initial_slip_resistance*/);

  tmp_slip_resistances      = slip_resistances;

  flag_init_was_called      = true;
}



template <int dim>
void QuadraturePointHistory<dim>::store_current_values()
{
  tmp_slip_resistances = slip_resistances;
}



template <int dim>
void QuadraturePointHistory<dim>::update_values(
  const unsigned int                      q_point,
  const std::vector<std::vector<double>>  &slips,
  const std::vector<std::vector<double>>  &old_slips)
{
  slip_resistances = tmp_slip_resistances;

  for (unsigned int slip_id_alpha = 0;
        slip_id_alpha < n_slips;
        ++slip_id_alpha)
    for (unsigned int slip_id_beta = 0;
          slip_id_beta < n_slips;
          ++slip_id_beta)
      slip_resistances[slip_id_alpha] +=
        get_hardening_matrix_entry(slip_id_alpha == slip_id_beta) *
        std::fabs(slips[slip_id_beta][q_point] -
                  old_slips[slip_id_beta][q_point]);
}



template <typename CellIteratorType, typename DataType>
std::vector<std::shared_ptr<DataType>>
InterfaceDataStorage<CellIteratorType, DataType>::get_data(
  const dealii::CellId current_cell_id,
  const dealii::CellId neighbour_cell_id)
{
  std::pair<dealii::CellId, dealii::CellId> key;

  if (current_cell_id < neighbour_cell_id)
    key =
      std::make_pair(current_cell_id, neighbour_cell_id);
  else
    key =
      std::make_pair(neighbour_cell_id, current_cell_id);

  Assert(map.find(key) != map.end(),
         dealii::ExcMessage(
           "The dealii::CellId pair does not correspond "
           "to a pair at the interface."));

  return map[key];
}


template <typename CellIteratorType, typename DataType>
std::vector<std::shared_ptr<const DataType>>
InterfaceDataStorage<CellIteratorType, DataType>::get_data(
  const dealii::CellId current_cell_id,
  const dealii::CellId neighbour_cell_id) const
{
  std::pair<dealii::CellId, dealii::CellId> key;

  if (current_cell_id < neighbour_cell_id)
    key =
      std::make_pair(current_cell_id, neighbour_cell_id);
  else
    key =
      std::make_pair(neighbour_cell_id, current_cell_id);

  Assert(map.find(key) != map.end(),
         dealii::ExcMessage(
           "The dealii::CellId pair does not correspond "
           "to a pair at the interface."));

  const auto it = map.find(key);


  const unsigned int n_face_q_points = it->second.size();

  std::vector<std::shared_ptr<const DataType>> tmp(n_face_q_points);

  for (unsigned int face_q_point = 0;
        face_q_point < n_face_q_points; ++face_q_point)
    tmp[face_q_point] =
      std::dynamic_pointer_cast<const DataType>(it->second[face_q_point]);

  return tmp;
}




} // namespace gCP



template class
gCP::InterfaceData<2>;
template class
gCP::InterfaceData<3>;

template class
gCP::InterfaceQuadraturePointHistory<2>;
template class
gCP::InterfaceQuadraturePointHistory<3>;

template gCP::QuadraturePointHistory<2>::QuadraturePointHistory();
template gCP::QuadraturePointHistory<3>::QuadraturePointHistory();

template void gCP::QuadraturePointHistory<2>::store_current_values();
template void gCP::QuadraturePointHistory<3>::store_current_values();

template void gCP::QuadraturePointHistory<2>::init(
  const RunTimeParameters::ScalarMicroscopicStressLawParameters &,
  const unsigned int                                            );
template void gCP::QuadraturePointHistory<3>::init(
  const RunTimeParameters::ScalarMicroscopicStressLawParameters &,
  const unsigned int                                            );

template void gCP::QuadraturePointHistory<2>::update_values(
  const unsigned int,
  const std::vector<std::vector<double>>  &,
  const std::vector<std::vector<double>>  &);
template void gCP::QuadraturePointHistory<3>::update_values(
  const unsigned int,
  const std::vector<std::vector<double>>  &,
  const std::vector<std::vector<double>>  &);

template class
gCP::InterfaceDataStorage<
  typename dealii::Triangulation<2>::cell_iterator,
  gCP::InterfaceData<2>>;
template class
gCP::InterfaceDataStorage<
  typename dealii::Triangulation<3>::cell_iterator,
  gCP::InterfaceData<3>>;

template class
gCP::InterfaceDataStorage<
  typename dealii::Triangulation<2>::cell_iterator,
  gCP::InterfaceQuadraturePointHistory<2>>;
template class
gCP::InterfaceDataStorage<
  typename dealii::Triangulation<3>::cell_iterator,
  gCP::InterfaceQuadraturePointHistory<3>>;



