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



enum class ControlType
{
  Displacement,

  Load,
};



enum class DamageEvolutionModel
{
  OrtizEtAl,

  M1
};



enum class CohesiveLawModel
{
  OrtizEtAl,
};



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

  CyclicWithUnloading,
};



/*!
 * @brief A enum class specifiying the type of solver
 */
enum class SolverType
{
  /*!
   * @brief Trilinos' direct solver
   *
   * @note Only the default parameters are implemented
   */
  DirectSolver,

  /*!
   * @brief Trilinos' conjugated gradient solver.
   *
   * @note Only the default parameters are implemented
   */
  CG,

  /*!
   * @brief Trilinos' generalized minimal residual method solver.
   *
   * @note Only the default parameters are implemented
   */
  GMRES,
};



enum class SolutionAlgorithm
{
  Monolithic,

  Bouncing,

  Embracing
};



enum class MonolithicPreconditioner
{
  BuiltIn,

  Block
};



/*!
 * @brief Enum listing all the implemented regularizations of the sign
 * function
 *
 * @details The approximations are controlled by the regularization
 * paramter \f$ k \f$
 */
enum class RegularizationFunction
{
  /*!
   * @brief The arctangent function approximation
   *
   * @details Defined as
   *
   * \f[
   *    \sgn(x) \approx  \frac{2}{\pi} \atan \left(\frac{\pi}{2} \frac{x}{k} \right)
   * \f]
   */
  Atan,

  /*!
   * @brief The square root approximation
   *
   * @details Defined as
   *
   * \f[
   *    \sgn(x) \approx \frac{x}{\sqrt{x^2 + k^2}}
   * \f]
   */
  Sqrt,


  /*!
   * @brief The Gudermannian function approximation
   *
   * @details Defined as
   *
   * \f[
   *    \sgn(x) \approx \frac{2}{\pi} \gd \left( \frac{\pi}{2} \frac{x}{k} \right)
   *    = \frac{2}{\pi} \atan \left( \sinh \left( \frac{\pi}{2} \frac{x}{k} \right) \right)
   * \f]
   */
  Gd,

  /*!
   * @brief The hyperbolic tangent approximation
   *
   * @details Defined as
   *
   * \f[
   *    \sgn(x) \approx \tanh \left( \frac{x}{k} \right)
   * \f]
   */
  Tanh,

  /*!
   * @brief The error function approximation
   *
   * @details Defined as
   *
   * \f[
   *    \sgn(x) \approx \erf  \left( \frac{\sqrt{\pi}}{2} \frac{x}{k} \right)
   * \f]
   */
  Erf,
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



struct ScalarMicrostressLawParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  ScalarMicrostressLawParameters();

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
  bool                    flag_rate_independent;
};



struct VectorialMicrostressLawParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  VectorialMicrostressLawParameters();

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

  /*!
   * @brief
   *
   * @todo Docu
   */
  double                  defect_energy_index;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double                  regularization_parameter;
};



struct MicrotractionLawParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  MicrotractionLawParameters();

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
  CohesiveLawModel  cohesive_law_model;

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
};




struct DegradationFunction
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  DegradationFunction();

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
  double  degradation_exponent;
};



struct HardeningLaw
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  HardeningLaw();

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

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool                    flag_perfect_plasticity;
};



struct DamageEvolution
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  DamageEvolution();

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
  DamageEvolutionModel damage_evolution_model;

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



/*!
 * @brief Struct containing all relevant parameters of the contact law
 * @details See @ref ConstitutiveLaws::ContactLaw for explanation of
 * each member
 */
struct ContactLawParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  ContactLawParameters();

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
   * @brief See @ref ConstitutiveLaws::ContactLaw
   */
  double  penalty_coefficient;
};


struct ConstitutiveLawsParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  ConstitutiveLawsParameters();

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
  HookeLawParameters    hooke_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  ScalarMicrostressLawParameters
                        scalar_microstress_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  VectorialMicrostressLawParameters
                        vectorial_microstress_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  MicrotractionLawParameters
                        microtraction_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  CohesiveLawParameters cohesive_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  ContactLawParameters  contact_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  HardeningLaw          hardening_law_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  DegradationFunction   degradation_function_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  DamageEvolution       damage_evolution_parameters;
};



struct CharacteristicQuantities
{
  CharacteristicQuantities();

  double length;

  double time;

  double displacement;

  double stiffness;

  double slip_resistance;

  double strain;

  double stress;

  double resolved_shear_stress;

  double macro_traction;

  double micro_traction;

  double body_force;

  double dislocation_density;
};



struct DimensionlessForm
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  DimensionlessForm();

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
   * @param prm
   * @todo Docu
   */
  void init(const RunTimeParameters::ConstitutiveLawsParameters &prm);

  CharacteristicQuantities characteristic_quantities;

  std::vector<double> dimensionless_numbers;

  bool flag_solve_dimensionless_problem;
};



struct KrylovParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  KrylovParameters();

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
  SolverType    solver_type;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double        relative_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double        absolute_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  unsigned int  n_max_iterations;
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
  double        relative_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double        absolute_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double        step_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  unsigned int  n_max_iterations;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool          flag_line_search;
};



struct LineSearchParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  LineSearchParameters();

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
  double        alpha;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double        beta;

  /*!
   * @brief
   *
   * @todo Docu
   */
  unsigned int  n_max_iterations;
};



struct NonlinearSystemSolverParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  NonlinearSystemSolverParameters();

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
  KrylovParameters              krylov_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  NewtonRaphsonParameters       newton_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  LineSearchParameters          line_search_parameters;
};




struct MonolithicAlgorithmParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  MonolithicAlgorithmParameters();

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
  NonlinearSystemSolverParameters monolithic_system_solver_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  MonolithicPreconditioner monolithic_preconditioner;
};



struct StaggeredAlgorithmParameters
{
  /*
   * @brief Constructor which sets up the parameters with default values.
   */
  StaggeredAlgorithmParameters();

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
  NonlinearSystemSolverParameters linear_momentum_solver_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  NonlinearSystemSolverParameters pseudo_balance_solver_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  unsigned int max_n_solution_loops;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool flag_reset_trial_solution_at_micro_loop;
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
  SolutionAlgorithm             solution_algorithm;

  /*!
   * @brief
   *
   * @todo Docu
   */
  MonolithicAlgorithmParameters monolithic_algorithm_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  StaggeredAlgorithmParameters staggered_algorithm_parameters;

  /*!
   * @brief
   *f
   * @todo Docu
   */
  ConstitutiveLawsParameters    constitutive_laws_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  DimensionlessForm
                                dimensionless_form_parameters;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool                          allow_decohesion;

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
  std::string                   logger_output_directory;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool                          flag_skip_extrapolation;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool                          flag_skip_extrapolation_at_extrema;

    /*!
   * @brief
   *
   * @todo Docu
   */
  double                        extrapolation_factor;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool                          flag_zero_damage_during_loading_and_unloading;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool                          flag_output_debug_fields;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool                          print_sparsity_pattern;

  /*!
   * @brief
   *
   * @todo Docu
   */
  bool                          verbose;
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
};



struct SimpleLoading : public TemporalDiscretizationParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  SimpleLoading();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  SimpleLoading(const std::string &parameter_filename);

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
   * @brief Get the next time step size object
   *
   * @param time
   * @return double
   * @todo Docu
   */
  double get_next_time_step_size(const unsigned int step_number) const;

  /*!
   * @brief
   *
   * @param step_number
   * @return true
   * @return false
   * @todo Docu
   */
  bool skip_extrapolation(const unsigned int step_number) const;

  LoadingType   loading_type;


  double        max_load;

  double        min_load;


  double        duration_monotonic_load;

  unsigned int  n_steps_monotonic_load;

  double        time_step_size_monotonic_load;


  double        duration_loading_and_unloading_phase;

  unsigned int  n_steps_loading_and_unloading_phase;

  double        time_step_size_loading_and_unloading_phase;


  unsigned int  n_cycles;

  double        period;

  unsigned int  n_steps_quarter_period;

  double        time_step_size_cyclic_phase;

  bool          flag_skip_unloading_phase;

//private:

  double        start_of_cyclic_phase;

  double        start_of_unloading_phase;
};



struct SpatialDiscretizationBase
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  SpatialDiscretizationBase();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  SpatialDiscretizationBase(const std::string &parameter_filename);

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

  unsigned int  dim;

  unsigned int  fe_degree_displacements;

  unsigned int  fe_degree_slips;

  unsigned int  n_global_refinements;

  unsigned int  mapping_degree;

  bool          flag_apply_mapping_to_interior_cells;
};



struct Input
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  Input();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  Input(const std::string &parameter_filename);

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

  std::string slips_normals_pathname;

  std::string slips_directions_pathname;

  std::string euler_angles_pathname;
};



struct Output
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  Output();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  Output(const std::string &parameter_filename);

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

  std::string   output_directory;

  unsigned int  graphical_output_frequency;

  unsigned int  terminal_output_frequency;

  unsigned int  homogenization_output_frequency;

  bool          flag_output_damage_variable;

  bool          flag_output_fluctuations;

  bool          flag_output_dimensionless_quantities;

  bool          flag_store_checkpoint;
};



struct Homogenization
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  Homogenization();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  Homogenization(const std::string &parameter_filename);

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

  unsigned int  homogenization_frequency;

  bool          flag_compute_homogenized_quantities;
};



struct BasicProblem
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  BasicProblem();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  BasicProblem(const std::string &parameter_filename);

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

  SpatialDiscretizationBase         spatial_discretization;

  TemporalDiscretizationParameters  temporal_discretization_parameters;

  SolverParameters                  solver_parameters;

  Input                             input;

  Output                            output;

  Homogenization                    homogenization;

  bool                              verbose;
};



struct InfiniteStripProblem : public BasicProblem
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  InfiniteStripProblem();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  InfiniteStripProblem(const std::string &parameter_filename);

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

  SimpleLoading simple_loading;

  ControlType   control_type;

  double        height;

  unsigned int  n_elements_in_y_direction;

  unsigned int  n_equal_sized_crystals;
};



struct RVEProblem : public BasicProblem
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  RVEProblem();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  RVEProblem(const std::string &parameter_filename);

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

  //RVEBoundaryConditionType  control_type;

  //StrainLoading             strain_loading;

  std::string               mesh_file_pathname;
};



struct SemicoupledParameters : public BasicProblem
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  SemicoupledParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  SemicoupledParameters(const std::string &parameter_filename);

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

  double      strain_component_11;

  double      strain_component_22;

  double      strain_component_33;

  double      strain_component_23;

  double      strain_component_13;

  double      strain_component_12;

  double      min_to_max_strain_load_ratio;

  std::string msh_file_pathname;
};


/*
template<typename Stream>
Stream& operator<<(Stream &stream, const Parameters &prm);
*/

}  // RunTimeParameters



}  // namespace gCP



#endif /* INCLUDE_RUN_TIME_PARAMETERS_H_ */
