struct RunTimeParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  RunTimeParameters();

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
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at
   * the end.
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const RunTimeParameters &prm);
};



template<int dim>
class NewtonRaphsonMethod
{
public:
  NewtonRaphsonMethod();
private:

  double                        absolute_tolerance;

  double                        relative_tolerance;

  double                        relaxation_parameter;

  double                        norm_residuum;

  dealii::SparseMatrix<double>  jacobian_matrix;

  dealii::Vector<double>        solution_increment;

  dealii::Vector<double>        residuum;
};
