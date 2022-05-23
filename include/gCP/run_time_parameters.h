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
  unsigned int  n_maximum_iterations;
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
  double              relative_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  double              absolute_tolerance;

  /*!
   * @brief
   *
   * @todo Docu
   */
  unsigned int        n_maximum_iterations;

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

  unsigned int            fe_degree_displacements;

  unsigned int            fe_degree_slips;

  RegularizationFunction  regularization_function;

  SolverParameters        solver_parameters;

  unsigned int            graphical_output_frequency;

  unsigned int            terminal_output_frequency;

  std::string             graphical_output_directory;

  bool                    verbose;
};



/*
template<typename Stream>
Stream& operator<<(Stream &stream, const Parameters &prm);
*/

}  // RunTimeParameters



}  // namespace gCP



#endif /* INCLUDE_RUN_TIME_PARAMETERS_H_ */