#ifndef INCLUDE_POINT_HISTORY_H_
#define INCLUDE_POINT_HISTORY_H_

#include <gCP/run_time_parameters.h>

#include <deal.II/distributed/tria.h>

#include <map>

namespace gCP
{



/*!
 * @brief
 *
 * @tparam dim
 * @todo Docu
 */
template <int dim>
class InterfaceData
{
public:

  InterfaceData() = default;

  virtual ~InterfaceData() = default;

  double get_value() const;

  void init(const double value);

  void prepare_for_update_call();

  void update(const dealii::Tensor<1,dim> a,
              const dealii::Tensor<1,dim> b);

private:

  double value;

  bool   was_updated;
};



template <int dim>
inline double InterfaceData<dim>::get_value() const
{
  return (value);
}



/*!
 * @brief
 *
 * @tparam dim
 * @todo Docu
 */
template <int dim>
class InterfaceQuadraturePointHistory
{
public:

  InterfaceQuadraturePointHistory();

  virtual ~InterfaceQuadraturePointHistory() = default;

  double get_damage_variable() const;

  double get_max_effective_opening_displacement() const;

  void init(
    const RunTimeParameters::DamageEvolution        &parameters,
    const RunTimeParameters::CohesiveLawParameters  &prm);

  void set(const double damage_variable_value);

  void store_current_values();

  void reset_values();

  void update_values(
    const double effective_opening_displacement,
    const double characterstic_displacement,
    const double thermodynamic_force);

  void update_values(
    const double  effective_opening_displacement);

  /*!
   * @brief Computes the effective opening displacement and stores it
   * in @ref old_effective_opening_displacement.
   *
   * @details This method is to be called at the end of each pseudo-time
   * iteration. It is needed as the effective opening displacement is
   * dependent of the normal vector when the material is anisotropic,
   * i.e., @ref tangential_to_normal_stiffness_ratio is different than
   * zero
   *
   * @param neighbor_cell_displacement
   * @param current_cell_displacement
   * @param normal_vector
   * @todo Docu
   */
  void store_effective_opening_displacement(
    const dealii::Tensor<1,dim> neighbor_cell_displacement,
    const dealii::Tensor<1,dim> current_cell_displacement,
    const dealii::Tensor<1,dim> normal_vector);

  /*!
   * @brief Temporary method
   */
  void store_effective_opening_displacement(
    const dealii::Tensor<1,dim> neighbor_cell_displacement,
    const dealii::Tensor<1,dim> current_cell_displacement,
    const dealii::Tensor<1,dim> normal_vector,
    const double                effective_cohesive_traction);

  // The methods
  double get_old_effective_opening_displacement() const;

  double get_effective_cohesive_traction() const;

  double get_effective_opening_displacement() const;

  // are temporary and will be deleted eventually.

private:
  double                    critical_cohesive_traction;

  double                    critical_opening_displacement;

  double                    tangential_to_normal_stiffness_ratio;

  double                    damage_accumulation_constant;

  double                    damage_decay_constant;

  double                    damage_decay_exponent;

  double                    endurance_limit;

  double                    damage_variable;

  double                    max_effective_opening_displacement;

  double                    old_effective_opening_displacement;

  std::vector<double>       tmp_scalar_values;

  // The variables
  double                    effective_opening_displacement;

  double                    normal_opening_displacement;

  double                    tangential_opening_displacement;

  double                    effective_cohesive_traction;

  // are temporary and will be deleted eventually.

  bool                      flag_set_damage_to_zero;

  bool                      flag_values_were_updated;

  bool                      flag_init_was_called;

  double macaulay_brackets(const double value) const;

  double get_master_relation(
    const double effective_opening_displacement) const;
};



template <int dim>
inline double InterfaceQuadraturePointHistory<dim>::
get_damage_variable() const
{
  return (damage_variable);
}



template <int dim>
inline double InterfaceQuadraturePointHistory<dim>::
get_max_effective_opening_displacement() const
{
  return (max_effective_opening_displacement);
}



template <int dim>
inline double InterfaceQuadraturePointHistory<dim>::
get_old_effective_opening_displacement() const
{
  return (old_effective_opening_displacement);
}



template <int dim>
inline double InterfaceQuadraturePointHistory<dim>::
get_effective_cohesive_traction() const
{
  return (effective_cohesive_traction);
}



template <int dim>
inline double InterfaceQuadraturePointHistory<dim>::
get_effective_opening_displacement() const
{
  return (effective_opening_displacement);
}



template <int dim>
inline double
InterfaceQuadraturePointHistory<dim>::macaulay_brackets(
  const double value) const
{
  if (value > 0)
    return value;
  else
    return 0.0;
}



template <int dim>
inline double
InterfaceQuadraturePointHistory<dim>::get_master_relation(
  const double effective_opening_displacement) const
{
  return (critical_cohesive_traction *
          effective_opening_displacement /
          critical_opening_displacement *
          std::exp(1.0 - effective_opening_displacement /
                         critical_opening_displacement));
}



template <int dim>
struct QuadraturePointHistory
{
public:
  /*!
   * @brief Default constructor
   *
   */
  QuadraturePointHistory();

  /*!
   * @brief Default destructor
   *
   */
  virtual ~QuadraturePointHistory() = default;

  /*!
   * @brief Get the slip resistance value at the quadrature point
   * of a slip system
   *
   * @param slip_id The id number of the slip system
   * @return double The slip resistance value corresponding to
   * @ref slip_id
   */
  double get_slip_resistance(const unsigned int slip_id) const;

  /*!
   * @brief Get the old slip resistance value at the quadrature point
   * of a slip system
   *
   * @param slip_id The id number of the slip system
   * @return double The slip resistance value corresponding to
   * @ref slip_id
   */
  double get_old_slip_resistance(const unsigned int slip_id) const;

  /*!
   * @brief Get the slip resistance value at the quadrature point
   * of all slip systems
   *
   * @return std::vector<double> The slip resistance value of all
   * slip systems
   */
  std::vector<double> get_slip_resistances() const;
  /*!
   * @brief Get the old_slip resistance value at the quadrature point
   * of all slip systems
   *
   * @return std::vector<double> The slip resistance value of all
   * slip systems
   */
  std::vector<double> get_old_slip_resistances() const;

  /*!
   * @brief Initiates the @ref QuadraturePointHistory instance
   *
   * @param parameters The material parameters of the evolution
   * equation
   * @param n_slips The number of slip systems of the crystal
   */
  void init(
    const RunTimeParameters::HardeningLaw &parameters,
    const unsigned int                    n_slips,
    const double characteristic_slip_resistance = 1.0);

  /*!
   * @brief Stores the values of @ref slip_resistances in @ref
   * tmp_slip_resistances
   *
   * @details The values are stored in order to reset
   * @ref slip_resistances in @ref update_values
   */
  void store_current_values();

  /*!
   * @brief Resets the values of @ref slip_resistances back to the value
   * stored in @ref tmp_slip_resistances
   */
  void reset_values();

  /*!
   * @brief Updates the slip resistance values at the quadratue point
   *
   * @details The slip resitance values at the quadrature point @ref
   * q_point are computed using the temporally discretized evolution
   * equation
   *
   * \f[
   *    g^{n}_{\alpha} =
   *     g^{n-1}_{\alpha} +
   *     \sum_1^{\text{n_slips}} h_{\alpha\beta}
   *      \abs{\gamma^{n}_{\alpha} - \gamma^{n-1}_{\alpha}}
   * \f]
   *
   * @param q_point The quadrature point at which the slip resitance
   * values are updated
   * @param slips The slip values at t^{n}
   * @param old_slips The slip values at t^{n-1}
   * @todo Docu
   */
  void update_values(
    const unsigned int                      q_point,
    const std::vector<std::vector<double>>  &slips,
    const std::vector<std::vector<double>>  &old_slips);

private:
  unsigned int        n_slips;

  double              initial_slip_resistance;

  std::vector<double> slip_resistances;

  std::vector<double> tmp_slip_resistances;

  double              linear_hardening_modulus;

  double              hardening_parameter;

  double              characteristic_slip_resistance;

  bool                flag_perfect_plasticity;

  bool                flag_init_was_called;

  double get_hardening_matrix_entry(const bool self_hardening) const;
};



template <int dim>
inline double
QuadraturePointHistory<dim>::get_slip_resistance(
  const unsigned int slip_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  return (slip_resistances[slip_id]);
}


template <int dim>
inline double
QuadraturePointHistory<dim>::get_old_slip_resistance(
  const unsigned int slip_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  return (tmp_slip_resistances[slip_id]);
}



template <int dim>
inline std::vector<double>
QuadraturePointHistory<dim>::get_slip_resistances() const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  return (slip_resistances);
}



template <int dim>
inline std::vector<double>
QuadraturePointHistory<dim>::get_old_slip_resistances() const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  return (tmp_slip_resistances);
}



template <int dim>
inline double
QuadraturePointHistory<dim>::get_hardening_matrix_entry(
  const bool self_hardening) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  return (linear_hardening_modulus /
          characteristic_slip_resistance *
          (hardening_parameter +
           ((self_hardening) ? (1.0 - hardening_parameter) : 0.0)));
}



/*!
 * @brief
 *
 * @tparam DataType
 * @tparam dim
 * @todo Docu
 */
template <typename CellIteratorType, typename DataType>
class InterfaceDataStorage : public dealii::Subscriptor
{
public:
  /*!
   * @brief Default constructor
   */
  InterfaceDataStorage() = default;

  /*!
   * @brief Default destructor
   */
  ~InterfaceDataStorage() override = default;

  /*!
   * @brief
   *
   * @param cell_start
   * @param cell_end
   * @param n_q_points_per_face
   * @todo Docu
   */
  void initialize(
    const CellIteratorType  &cell_start,
    const CellIteratorType  &cell_end,
    const unsigned int      n_q_points_per_face);

  /*!
   * @brief Get the data object
   *
   * @param current_cell_id
   * @param neighbor_cell_id
   * @return std::vector<std::shared_ptr<DataType>>
   * @todo Docu
   */
  std::vector<std::shared_ptr<DataType>>
    get_data(const dealii::CellId current_cell_id,
             const dealii::CellId neighbor_cell_id);

  /*!
   * @brief Get the data object
   *
   * @param current_cell_id
   * @param neighbor_cell_id
   * @return std::vector<std::shared_ptr<const DataType>>
   * @warning Compilation error when used
   * @todo Docu
   */
  std::vector<std::shared_ptr<const DataType>>
    get_data(const dealii::CellId current_cell_id,
             const dealii::CellId neighbor_cell_id) const;

private:
  /**
   * Number of dimensions
   */
  static constexpr unsigned int dimension =
    CellIteratorType::AccessorType::dimension;

  /**
   * Number of space dimensions
   */
  static constexpr unsigned int space_dimension =
    CellIteratorType::AccessorType::space_dimension;

  /**
   * To ensure that all the cells in the CellDataStorage come from the
   * same Triangulation, we need to store a reference to that
   * Triangulation within the class.
   */
  dealii::SmartPointer<const dealii::Triangulation<dimension, dimension>,
               InterfaceDataStorage<CellIteratorType, DataType>>
    tria;

  /**
   * A map to store a vector of data on each cell.
   * We need to use CellId as the key because it remains unique during
   * adaptive refinement.
   */
  std::map<
    std::pair<dealii::CellId, dealii::CellId>,
    std::vector<std::shared_ptr<DataType>>> map;

  DeclExceptionMsg(
    ExcTriangulationMismatch,
    "The provided cell iterator does not belong to the triangulation that corresponds to the CellDataStorage object.");
};



template <typename CellIteratorType, typename DataType>
void InterfaceDataStorage<CellIteratorType, DataType>::initialize(
  const CellIteratorType  &cell_start,
  const CellIteratorType  &cell_end,
  const unsigned int      n_face_q_points)
{
  Assert(n_face_q_points > 0,
         dealii::ExcMessage(
           "The number of quadrature points per face has to be bigger "
           "than zero."));

  for (CellIteratorType cell = cell_start; cell != cell_end; ++cell)
    if (cell->is_locally_owned())
      for (const auto &face_index : cell->face_indices())
        if (!cell->face(face_index)->at_boundary() &&
            cell->material_id() !=
              cell->neighbor(face_index)->material_id())
        {
          // The first time this method is called, it has to initialize the reference
          // to the triangulation object
          if (!tria)
            tria = &cell->get_triangulation();
          Assert(&cell->get_triangulation() == tria,
                 ExcTriangulationMismatch());

          std::pair<dealii::CellId, dealii::CellId> key;

          const dealii::CellId current_cell_id =
            cell->id();
          const dealii::CellId neighbor_cell_id =
            cell->neighbor(face_index)->id();

          if (current_cell_id < neighbor_cell_id)
            key = std::make_pair(current_cell_id, neighbor_cell_id);
          else
            key = std::make_pair(neighbor_cell_id, current_cell_id);

          if (map.find(key) == map.end())
          {
            map[key] =
              std::vector<std::shared_ptr<DataType>>(n_face_q_points);

            for (unsigned int face_q_point = 0;
                 face_q_point < n_face_q_points; ++face_q_point)
              map[key][face_q_point] = std::make_shared<DataType>();
          }
        }
}



} // namespace gCP



#endif /* INCLUDE_POINT_HISTORY_H_ */
