#ifndef INCLUDE_RUN_TIME_PARAMETERS_H_
#define INCLUDE_RUN_TIME_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

namespace gCP
{


/*!
 * @brief Namespace encompasing all the structs containing the suite's
 * parameters
 */
namespace RunTimeParameters
{



/*!
 * @brief A enum class specifiying the type of loading
 */
enum class LoadingType
{
  /*!
   * @brief Monotonic load
   */
  Monotonic,

  /*!
   * @brief Cyclic loading.
   *
   * @details The load is divided into a initial loading phase and a
   * cyclic loading phase
   */
  Cyclic,
};



/*!
 * @brief
 *
 * @todo Docu
 */
enum class RegularizationFunction
{
  /*!
   * @brief
   *
   * @todo Docu
   */
  PowerLaw,

  /*!
   * @brief
   *
   * @todo Docu
   */
  Tanh,
};



/*!
 * @brief
 *
 * @todo Docu
 */
enum class BoundaryConditionsAtGrainBoundaries
{
  /*!
   * @brief
   *
   * @todo Docu
   */
  Microhard,

  /*!
   * @brief
   *
   * @todo Docu
   */
  Microfree,

  /*!
   * @brief
   *
   * @todo Docu
   */
  Microtraction,
};



struct NewtonRaphsonParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  NewtonRaphsonParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief
   *
   * @todo Docu
   */
  double        nonlinear_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  unsigned int  n_max_nonlinear_iterations;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double        krylov_relative_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double        krylov_absolute_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  unsigned int  n_max_krylov_iterations;
};



struct HookeLawParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  HookeLawParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  C1111;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  C1122;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  C1212;
};



struct ScalarMicroscopicStressLawParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  ScalarMicroscopicStressLawParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief
   *
   * @todo Docu
   */
  RegularizationFunction  regularization_function;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double                  regularization_parameter;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double                  initial_slip_resistance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double                  linear_hardening_modulus;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double                  hardening_parameter;
};



struct VectorMicroscopicStressLawParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  VectorMicroscopicStressLawParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief
   *
   * @todo Docu
   */
  double                  energetic_length_scale;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double                  initial_slip_resistance;
};



struct MicroscopicTractionLawParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  MicroscopicTractionLawParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  grain_boundary_modulus;
};



struct CohesiveLawParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  CohesiveLawParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  critical_cohesive_traction;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  critical_opening_displacement;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  tangential_to_normal_stiffness_ratio;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  damage_accumulation_constant;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  damage_decay_constant;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  damage_decay_exponent;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  endurance_limit;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double  degradation_exponent;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool    flag_couple_microtraction_to_damage;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool    flag_couple_macrotraction_to_damage;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool    flag_set_damage_to_zero;
};



struct SolverParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  SolverParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief
   *
   * @todo Docu
   */
  double              residual_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double              newton_update_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  unsigned int        n_max_nonlinear_iterations;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double              krylov_relative_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double              krylov_absolute_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  unsigned int        n_max_krylov_iterations;

  /*!
   * @brief
   *
   * @todo Docu
   */
  HookeLawParameters  hooke_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  ScalarMicroscopicStressLawParameters
                      scalar_microscopic_stress_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  VectorMicroscopicStressLawParameters
                      vector_microscopic_stress_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  MicroscopicTractionLawParameters
                      microscopic_traction_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  CohesiveLawParameters
                      cohesive_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool                allow_decohesion;

  /*!
   * @brief
   *
   * @todo Docu
   */
  BoundaryConditionsAtGrainBoundaries
                      boundary_conditions_at_grain_boundaries;

  /*!
   * @brief
   *
   * @todo Docu
   */
  std::string         logger_output_directory;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool                print_sparsity_pattern;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool                verbose;
};


/*!
 * @brief A struct containing the parameters of the temporal
 * discretization
 */
struct TemporalDiscretizationParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  TemporalDiscretizationParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  TemporalDiscretizationParameters(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(dealii::ParameterHandler &prm);

  /*
  template<typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const Parameters &prm);
  */

  /*!
   * @brief The start time of the simulation
   */
  double      start_time;

  /*!
   * @brief The end time of the simulation
   */
  double      end_time;

  /*!
   * @brief The time step size used during the simulation
   */
  double      time_step_size;

  /*!
   * @brief The period of the cyclic load
   *
   * @note This member is only relevant if @ref loading_type
   * corresponds to @ref SimulationTimeControl::Cyclic
   */
  double      period;

  /*!
   * @brief The number of cycles to be simulated
   *
   * @note This member is only relevant if @ref loading_type
   * corresponds to @ref SimulationTimeControl::Cyclic
   */
  int         n_cycles;

  /*!
   * @brief The number of discrete points per half cycle at which
   * the quasi static problem will be solved
   *
   * @note This member is only relevant if @ref loading_type
   * corresponds to @ref SimulationTimeControl::Cyclic
   */
  int         n_steps_per_half_cycle;

  /*!
   * @brief The time in which the initial loading takes place
   *
   * @note This member is only relevant if @ref loading_type
   * corresponds to @ref SimulationTimeControl::Cyclic
   */
  double      initial_loading_time;

  /*!
   * @brief The time step used during the initial loading phase
   *
   * @note This member is only relevant if @ref loading_type
   * corresponds to @ref SimulationTimeControl::Cyclic
   */
  int         n_steps_in_loading_phase;

  /*!
   * @brief The time step used during the loading phase
   *
   * @details It is internally computed using @ref initial_loading_time
   * and @ref n_steps_in_loading_phase
   *
   * @note This member is only relevant if @ref loading_type
   * corresponds to @ref SimulationTimeControl::Cyclic
   */
  double      time_step_size_in_loading_phase;

  /*!
   * @brief The simulation time control to be used. See @ref
   * RunTimeParameters::SimulationTimeControl
   */
  LoadingType  loading_type;
};



struct ProblemParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ProblemParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  ProblemParameters(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(dealii::ParameterHandler &prm);

  /*
  template<typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const Parameters &prm);
  */

  unsigned int                      dim;

  unsigned int                      mapping_degree;

  bool                              mapping_interior_cells;

  unsigned int                      n_global_refinements;

  unsigned int                      fe_degree_displacements;

  unsigned int                      fe_degree_slips;

  TemporalDiscretizationParameters  temporal_discretization_parameters;

  SolverParameters                  solver_parameters;

  std::string                       slips_normals_pathname;

  std::string                       slips_directions_pathname;

  std::string                       euler_angles_pathname;

  unsigned int                      graphical_output_frequency;

  unsigned int                      terminal_output_frequency;

  std::string                       graphical_output_directory;

  bool                              verbose;
};



struct SimpleShearParameters : public ProblemParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  SimpleShearParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  SimpleShearParameters(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(dealii::ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(dealii::ParameterHandler &prm);

  double        max_shear_strain_at_upper_boundary;

  double        min_shear_strain_at_upper_boundary;

  unsigned int  n_equal_sized_divisions;

  double        height;

  double        width;
};


/*
template<typename Stream>
Stream& operator<<(Stream &stream, const Parameters &prm);
*/

}  // RunTimeParameters



}  // namespace gCP



#endif /* INCLUDE_RUN_TIME_PARAMETERS_H_ */