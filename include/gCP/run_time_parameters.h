#ifndef INCLUDE_RUN_TIME_PARAMETERS_H_
#define INCLUDE_RUN_TIME_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

namespace gCP
{



namespace RunTimeParameters
{


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

  unsigned int            dim;

  unsigned int            mapping_degree;

  bool                    mapping_interior_cells;

  unsigned int            n_global_refinements;

  double                  start_time;

  double                  end_time;

  double                  time_step_size;

  unsigned int            fe_degree_displacements;

  unsigned int            fe_degree_slips;

  SolverParameters        solver_parameters;

  std::string             slips_normals_pathname;

  std::string             slips_directions_pathname;

  std::string             euler_angles_pathname;

  unsigned int            graphical_output_frequency;

  unsigned int            terminal_output_frequency;

  std::string             graphical_output_directory;

  bool                    verbose;
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

  double  shear_strain_at_upper_boundary;

  double  height;

  double  width;
};


/*
template<typename Stream>
Stream& operator<<(Stream &stream, const Parameters &prm);
*/

}  // RunTimeParameters



}  // namespace gCP



#endif /* INCLUDE_RUN_TIME_PARAMETERS_H_ */