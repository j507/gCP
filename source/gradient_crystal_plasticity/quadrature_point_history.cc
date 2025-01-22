
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
tangential_to_normal_stiffness_ratio(1.0),
damage_accumulation_constant(1.0),
damage_decay_constant(0.0),
damage_decay_exponent(1.0),
endurance_limit(0.0),
damage_variable(0.0),
max_effective_opening_displacement(0.0),
old_effective_opening_displacement(0.0),
tmp_scalar_values(2),
effective_opening_displacement(0.0),
normal_opening_displacement(0.0),
tangential_opening_displacement(0.0),
effective_cohesive_traction(0.0),
flag_init_was_called(false)
{}



template <int dim>
void InterfaceQuadraturePointHistory<dim>::init(
  const RunTimeParameters::DamageEvolution        &parameters,
  const RunTimeParameters::CohesiveLawParameters  &prm)
{
  if (flag_init_was_called)
    return;

  critical_cohesive_traction =
    prm.critical_cohesive_traction;

  critical_opening_displacement =
    prm.critical_opening_displacement;

  tangential_to_normal_stiffness_ratio =
    prm.tangential_to_normal_stiffness_ratio;

  damage_accumulation_constant =
    parameters.damage_accumulation_constant;

  damage_decay_constant   = parameters.damage_decay_constant;

  damage_decay_exponent   = parameters.damage_decay_exponent;

  endurance_limit         = parameters.endurance_limit;

  flag_set_damage_to_zero = parameters.flag_set_damage_to_zero;

  flag_init_was_called    = true;
}



template <int dim>
void InterfaceQuadraturePointHistory<dim>::set(
  const double damage_variable_value)
{
  damage_variable = damage_variable_value;
}



template <int dim>
void InterfaceQuadraturePointHistory<dim>::store_current_values()
{
  tmp_scalar_values[0]  = damage_variable;
  tmp_scalar_values[1]  = max_effective_opening_displacement;

  flag_values_were_updated = false;
}



template <int dim>
void InterfaceQuadraturePointHistory<dim>::reset_values()
{
  damage_variable                     = tmp_scalar_values[0];

  max_effective_opening_displacement  = tmp_scalar_values[1];
}



template <int dim>
void InterfaceQuadraturePointHistory<dim>::update_values(
  const double  effective_opening_displacement)
{
  damage_variable                     = tmp_scalar_values[0];
  max_effective_opening_displacement  = tmp_scalar_values[1];

  max_effective_opening_displacement =
    std::max(max_effective_opening_displacement,
             effective_opening_displacement);

  const double displacement_ratio =
     max_effective_opening_displacement / critical_opening_displacement;

  damage_variable =
    1.0 - (1.0 + displacement_ratio) * std::exp(-displacement_ratio);

  if (flag_set_damage_to_zero)
    damage_variable = 0.0;

  flag_values_were_updated = true;
}



template <int dim>
void InterfaceQuadraturePointHistory<dim>::update_values(
  const double effective_opening_displacement,
  const double characteristic_displacement,
  const double thermodynamic_force)
{
  damage_variable                     = tmp_scalar_values[0];
  max_effective_opening_displacement  = tmp_scalar_values[1];

  max_effective_opening_displacement =
    std::max(max_effective_opening_displacement,
             effective_opening_displacement);

  damage_variable +=
    damage_accumulation_constant *
    characteristic_displacement *
    macaulay_brackets(effective_opening_displacement -
                      old_effective_opening_displacement) *
    std::pow(1.0 - damage_variable + damage_decay_constant,
     damage_decay_exponent) *
    (thermodynamic_force - endurance_limit);

  if (damage_variable > 1.0)
    damage_variable = 1.0;

  if (flag_set_damage_to_zero)
    damage_variable = 0.0;

  flag_values_were_updated = true;
}



template <int dim>
void InterfaceQuadraturePointHistory<dim>::store_effective_opening_displacement(
  const dealii::Tensor<1,dim> current_cell_displacement,
  const dealii::Tensor<1,dim> neighbor_cell_displacement,
  const dealii::Tensor<1,dim> normal_vector)
{
  /*if (flag_values_were_updated)
    return;*/

  // Define projectors
  dealii::SymmetricTensor<2,dim> normal_projector =
    dealii::symmetrize(dealii::outer_product(normal_vector,
                                             normal_vector));

  dealii::SymmetricTensor<2,dim> tangential_projector =
    dealii::unit_symmetric_tensor<dim>() - normal_projector;

  // Compute opening displacement
  dealii::Tensor<1,dim> opening_displacement =
    neighbor_cell_displacement - current_cell_displacement;

  // Compute normal and tangential components
  double normal_opening_displacement =
    (normal_projector * opening_displacement).norm();

  double tangential_opening_displacement =
    (tangential_projector * opening_displacement).norm();

  // Compute effective opening displacements (Needed for the next
  // pseudo-time iteration)
  old_effective_opening_displacement =
     std::sqrt(macaulay_brackets(normal_opening_displacement) *
               macaulay_brackets(normal_opening_displacement)
               +
               tangential_to_normal_stiffness_ratio *
               tangential_to_normal_stiffness_ratio *
               tangential_opening_displacement *
               tangential_opening_displacement);
}



template <int dim>
void InterfaceQuadraturePointHistory<dim>::store_effective_opening_displacement(
  const dealii::Tensor<1,dim> current_cell_displacement,
  const dealii::Tensor<1,dim> neighbor_cell_displacement,
  const dealii::Tensor<1,dim> normal_vector,
  const double                effective_cohesive_traction)
{
  /*if (flag_values_were_updated)
    return;*/

  // Define projectors
  dealii::SymmetricTensor<2,dim> normal_projector =
    dealii::symmetrize(dealii::outer_product(normal_vector,
                                             normal_vector));

  dealii::SymmetricTensor<2,dim> tangential_projector =
    dealii::unit_symmetric_tensor<dim>() - normal_projector;

  // Compute opening displacement
  dealii::Tensor<1,dim> opening_displacement =
    neighbor_cell_displacement - current_cell_displacement;

  // Compute normal and tangential components
  double normal_opening_displacement =
    (normal_projector * opening_displacement).norm();

  double tangential_opening_displacement =
    (tangential_projector * opening_displacement).norm();

  // Compute effective opening displacements (Needed for the next
  // pseudo-time iteration)
  old_effective_opening_displacement =
     std::sqrt(macaulay_brackets(normal_opening_displacement) *
               macaulay_brackets(normal_opening_displacement)
               +
               tangential_to_normal_stiffness_ratio *
               tangential_to_normal_stiffness_ratio *
               tangential_opening_displacement *
               tangential_opening_displacement);

  this->effective_opening_displacement = old_effective_opening_displacement;

  this->normal_opening_displacement = normal_opening_displacement;

  this->tangential_opening_displacement = tangential_opening_displacement;

  this->effective_cohesive_traction = effective_cohesive_traction;
}



template <int dim>
QuadraturePointHistory<dim>::QuadraturePointHistory()
:
flag_perfect_plasticity(false),
flag_init_was_called(false)
{}



template <int dim>
void QuadraturePointHistory<dim>::init(
  const RunTimeParameters::HardeningLaw &parameters,
  const unsigned int n_slips,
  const double characteristic_slip_resistance)
{
  this->n_slips             = n_slips;

  initial_slip_resistance   = parameters.initial_slip_resistance;

  linear_hardening_modulus  = parameters.linear_hardening_modulus;

  hardening_parameter       = parameters.hardening_parameter;

  flag_perfect_plasticity   = parameters.flag_perfect_plasticity;

  slip_resistances          = std::vector<double>(
                                n_slips,
                                initial_slip_resistance /
                                  characteristic_slip_resistance);

  tmp_slip_resistances      = slip_resistances;

  this->characteristic_slip_resistance =
    characteristic_slip_resistance;

  flag_init_was_called      = true;
}



template <int dim>
void QuadraturePointHistory<dim>::store_current_values()
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  tmp_slip_resistances = slip_resistances;
}



template <int dim>
void QuadraturePointHistory<dim>::reset_values()
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  slip_resistances = tmp_slip_resistances;
}



template <int dim>
void QuadraturePointHistory<dim>::update_values(
  const unsigned int                      q_point,
  const std::vector<std::vector<double>>  &slips,
  const std::vector<std::vector<double>>  &old_slips)
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  slip_resistances = tmp_slip_resistances;

  if (flag_perfect_plasticity)
  {
    return;
  }

  for (unsigned int slip_id_alpha = 0;
        slip_id_alpha < n_slips;
        ++slip_id_alpha)
  {
    for (unsigned int slip_id_beta = 0;
          slip_id_beta < n_slips;
          ++slip_id_beta)
    {
      slip_resistances[slip_id_alpha] +=
        get_hardening_matrix_entry(slip_id_alpha == slip_id_beta) *
        std::fabs(slips[slip_id_beta][q_point] -
                  old_slips[slip_id_beta][q_point]);
    }
  }
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

template class
gCP::QuadraturePointHistory<2>;
template class
gCP::QuadraturePointHistory<3>;


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



