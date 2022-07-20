#ifndef INCLUDE_POINT_HISTORY_H_
#define INCLUDE_POINT_HISTORY_H_

#include <gCP/run_time_parameters.h>

#include <deal.II/distributed/tria.h>


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
class InterfacialQuadraturePointHistory
{
public:

  InterfacialQuadraturePointHistory();

  virtual ~InterfacialQuadraturePointHistory() = default;

  double get_max_displacement_jump_norm() const;

  double get_damage_variable() const;

  void init(
    const RunTimeParameters::DecohesionLawParameters &parameters);

  void store_current_values();

  void update_values(
    const dealii::Tensor<1,dim> neighbor_cell_displacement,
    const dealii::Tensor<1,dim> current_cell_displacement);

private:
  double                    maximum_cohesive_traction;

  double                    critical_opening_displacement;

  //double                    critical_energy_release_rate;

  double                    max_displacement_jump_norm;

  double                    damage_variable;

  std::pair<double, double> tmp_values;

  bool                      flag_values_were_updated;

  bool                      flag_init_was_called;
};



template <int dim>
inline double InterfacialQuadraturePointHistory<dim>::
get_max_displacement_jump_norm() const
{
  return (max_displacement_jump_norm);
}



template <int dim>
inline double InterfacialQuadraturePointHistory<dim>::
get_damage_variable() const
{
  return (damage_variable);
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
   * @brief Get the slip resistance value at the quadrature point
   * of all slip systems
   *
   * @return std::vector<double> The slip resistance value of all
   * slip systems
   */
  std::vector<double> get_slip_resistances() const;

  /*!
   * @brief Initiates the @ref QuadraturePointHistory instance
   *
   * @param parameters The material parameters of the evolution
   * equation
   * @param n_slips The number of slip systems of the crystal
   */
  void init(
    const RunTimeParameters::ScalarMicroscopicStressLawParameters
      &parameters,
    const unsigned int n_slips);

  /*!
   * @brief Stores the values of @ref slip_resistances in @ref
   * tmp_slip_resistances
   *
   * @details The values are stored in order to reset
   * @ref slip_resistances in @ref update_values
   */
  void store_current_values();

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
inline std::vector<double>
QuadraturePointHistory<dim>::get_slip_resistances() const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  return (slip_resistances);
}



template <int dim>
inline double
QuadraturePointHistory<dim>::get_hardening_matrix_entry(
  const bool self_hardening) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  return (linear_hardening_modulus *
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
template <typename DataType, int dim>
class InterfaceDataStorage
{
public:
  /*!
   * @brief Default constructor
   */
  InterfaceDataStorage() = default;

  /*!
   * @brief Default destructor
   */
  ~InterfaceDataStorage() = default;

  /*!
   * @brief
   *
   * @param triangulation
   * @param n_q_points_per_face
   * @todo Docu
   */
  void initialize(
    const dealii::parallel::distributed::Triangulation<dim>
                                                  &triangulation,
    const unsigned int                            n_q_points_per_face);

  /*!
   * @brief Get the data object
   *
   * @param current_cell_id
   * @param neighbor_cell_id
   * @return std::vector<std::shared_ptr<DataType>>
   * @todo Docu
   */
  std::vector<std::shared_ptr<DataType>>
    get_data(const unsigned int current_cell_id,
             const unsigned int neighbor_cell_id);

  /*!
   * @brief Get the data object
   *
   * @param current_cell_id
   * @param neighbor_cell_id
   * @return std::vector<std::shared_ptr<const DataType>>
   * @warning Compilation error when used
   * @todo Docu
   */
  /*std::vector<std::shared_ptr<const DataType>>
    get_data(const unsigned int current_cell_id,
             const unsigned int neighbor_cell_id) const;*/

private:
  /*!
   * @brief
   *
   * @todo Docu
   */
  std::map<
    std::pair<unsigned int, unsigned int>,
    std::vector<std::shared_ptr<DataType>>> map;
};



} // namespace gCP



#endif /* INCLUDE_POINT_HISTORY_H_ */
