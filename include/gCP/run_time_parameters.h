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



struct Parameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  Parameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  Parameters(const std::string &parameter_filename);

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

  double                  relative_tolerance;

  double                  absolute_tolerance;

  unsigned int            n_maximum_iterations;

  HookeLawParameters      hooke_law_parameters;

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