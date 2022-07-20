
#include <gCP/quadrature_point_history.h>



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
InterfacialQuadraturePointHistory<dim>::InterfacialQuadraturePointHistory()
:
max_displacement_jump_norm(0.0),
damage_variable(0.0),
flag_init_was_called(false)
{}



template <int dim>
void InterfacialQuadraturePointHistory<dim>::init(
  const RunTimeParameters::DecohesionLawParameters &parameters)
{
  maximum_cohesive_traction     = parameters.maximum_cohesive_traction;

  critical_opening_displacement = parameters.critical_opening_displacement;

  /*critical_energy_release_rate  = maximum_cohesive_traction *
                                  critical_opening_displacement *
                                  std::exp(1.0);*/

  flag_init_was_called          = true;
}



template <int dim>
void InterfacialQuadraturePointHistory<dim>::store_current_values()
{
  tmp_values = std::make_pair(max_displacement_jump_norm,
                              damage_variable);

  flag_values_were_updated = false;
}



template <int dim>
void InterfacialQuadraturePointHistory<dim>::update_values(
  const dealii::Tensor<1,dim> neighbor_cell_displacement,
  const dealii::Tensor<1,dim> current_cell_displacement)
{
  if (flag_values_were_updated)
    return;

  max_displacement_jump_norm  = tmp_values.first;
  damage_variable             = tmp_values.second;

  const dealii::Tensor<1,dim> displacement_jump =
    neighbor_cell_displacement - current_cell_displacement;

  max_displacement_jump_norm =
    std::max(max_displacement_jump_norm,
             displacement_jump.norm());

  const double displacement_ratio =
     max_displacement_jump_norm / critical_opening_displacement;

  /*
  const double free_energy_density =
    std::exp(1.0) *
    maximum_cohesive_traction *
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



template <typename DataType, int dim>
void InterfaceDataStorage<DataType, dim>::initialize(
  const dealii::parallel::distributed::Triangulation<dim> &triangulation,
  const unsigned int                                      n_q_points_per_face)
{
  Assert(triangulation.n_global_active_cells() > 0,
         dealii::ExcMessage(
           "The triangulation instance seems to be empty as it has no "
           "active cells."));

  Assert(n_q_points_per_face > 0,
         dealii::ExcMessage(
           "The number of quadrature points per face has to be bigger "
           "than zero."));

  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &face_index : cell->face_indices())
        if (!cell->face(face_index)->at_boundary() &&
            cell->material_id() !=
              cell->neighbor(face_index)->material_id())
        {
          std::pair<unsigned int, unsigned int> cell_pair;

          if (cell->active_cell_index() <
              cell->neighbor(face_index)->active_cell_index())
            cell_pair =
              std::make_pair(
                cell->active_cell_index(),
                cell->neighbor(face_index)->active_cell_index());
          else
            cell_pair =
              std::make_pair(
                cell->neighbor(face_index)->active_cell_index(),
                cell->active_cell_index());

          if (map.find(cell_pair) == map.end())
          {
            std::vector<std::shared_ptr<DataType>>
              container(n_q_points_per_face);

            for (auto &element : container)
              element = std::make_shared<DataType>();

            map.insert({cell_pair, container});

            cell->face(face_index)->set_user_pointer(&map[cell_pair]);
          }
        }
}



template <typename DataType, int dim>
std::vector<std::shared_ptr<DataType>>
InterfaceDataStorage<DataType, dim>::get_data(
  const unsigned int current_cell_id,
  const unsigned int neighbour_cell_id)
{
  std::pair<unsigned int, unsigned int> cell_pair;

  if (current_cell_id < neighbour_cell_id)
    cell_pair =
      std::make_pair(current_cell_id, neighbour_cell_id);
  else
    cell_pair =
      std::make_pair(neighbour_cell_id, current_cell_id);

  Assert(map.find(cell_pair) != map.end(),
         dealii::ExcMessage(
           "The pair {" + std::to_string(cell_pair.first) + ", " +
           std::to_string(cell_pair.second) + "} does not correspond "
           "to a pair at the interface."));

  return map[cell_pair];
}


/*
template <typename DataType, int dim>
std::vector<std::shared_ptr<const DataType>>
InterfaceDataStorage<DataType, dim>::get_data(
  const unsigned int current_cell_id,
  const unsigned int neighbour_cell_id) const
{
  std::pair<unsigned int, unsigned int> cell_pair;

  if (current_cell_id < neighbour_cell_id)
    cell_pair =
      std::make_pair(current_cell_id, neighbour_cell_id);
  else
    cell_pair =
      std::make_pair(neighbour_cell_id, current_cell_id);

  Assert(map.find(cell_pair) == map.end(),
         dealii::ExcMessage(
           "The pair {" + std::to_string(cell_pair.first) + ", " +
           std::to_string(cell_pair.second) + "} does not correspond "
           "to a pair at the interface."));

  return map[cell_pair];
}*/



} // namespace gCP



template class
gCP::InterfaceData<2>;
template class
gCP::InterfaceData<3>;

template class
gCP::InterfacialQuadraturePointHistory<2>;
template class
gCP::InterfacialQuadraturePointHistory<3>;

template class
gCP::InterfaceDataStorage<gCP::InterfaceData<2>,2>;
template class
gCP::InterfaceDataStorage<gCP::InterfaceData<3>,3>;

template class
gCP::InterfaceDataStorage<gCP::InterfacialQuadraturePointHistory<2>,2>;
template class
gCP::InterfaceDataStorage<gCP::InterfacialQuadraturePointHistory<3>,3>;

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
