#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/fe/mapping_q.h>

namespace gCP
{



template <int dim>
GradientCrystalPlasticitySolver<dim>::GradientCrystalPlasticitySolver(
  const RunTimeParameters::SolverParameters         &parameters,
  dealii::DiscreteTime                              &discrete_time,
  std::shared_ptr<FEField<dim>>                     &fe_field,
  std::shared_ptr<CrystalsData<dim>>                &crystals_data,
  const std::shared_ptr<dealii::Mapping<dim>>       external_mapping,
  const std::shared_ptr<dealii::ConditionalOStream> external_pcout,
  const std::shared_ptr<dealii::TimerOutput>        external_timer,
  const RunTimeParameters::LoadingType              loading_type)
:
parameters(parameters),
discrete_time(discrete_time),
fe_field(fe_field),
crystals_data(crystals_data),
elastic_strain(
  std::make_shared<Kinematics::ElasticStrain<dim>>(
    crystals_data)),
hooke_law(
  std::make_shared<ConstitutiveLaws::HookeLaw<dim>>(
    crystals_data,
    parameters.hooke_law_parameters)),
resolved_shear_stress_law(
  std::make_shared<ConstitutiveLaws::ResolvedShearStressLaw<dim>>(
    crystals_data)),
scalar_microscopic_stress_law(
  std::make_shared<ConstitutiveLaws::ScalarMicroscopicStressLaw<dim>>(
    crystals_data,
    parameters.scalar_microscopic_stress_law_parameters)),
vector_microscopic_stress_law(
  std::make_shared<ConstitutiveLaws::VectorMicroscopicStressLaw<dim>>(
    crystals_data,
    parameters.vector_microscopic_stress_law_parameters)),
microscopic_traction_law(
  std::make_shared<ConstitutiveLaws::MicroscopicTractionLaw<dim>>(
    crystals_data,
    parameters.microscopic_traction_law_parameters)),
cohesive_law(
  std::make_shared<ConstitutiveLaws::CohesiveLaw<dim>>(
    parameters.cohesive_law_parameters)),
residual_norm(std::numeric_limits<double>::max()),
nonlinear_solver_logger(parameters.logger_output_directory +
                        "nonlinear_solver_log.txt"),
loading_type(loading_type),
flag_init_was_called(false)
{
  Assert(fe_field.get() != nullptr,
         dealii::ExcMessage("The FEField<dim>'s shared pointer has "
                            "contains a nullptr."));
  Assert(crystals_data.get() != nullptr,
         dealii::ExcMessage("The CrystalsData<dim>'s shared pointer "
                            "contains a nullptr."));

  // Initiating the internal Mapping instances.
  if (external_mapping.get() != nullptr)
    mapping = external_mapping;
  else
    mapping = std::make_shared<dealii::MappingQ<dim>>(1);

  mapping_collection =
    dealii::hp::MappingCollection<dim>(*mapping);

  // Initiating the internal ConditionalOStream instance.
  if (external_pcout.get() != nullptr)
    pcout = external_pcout;
  else
    pcout = std::make_shared<dealii::ConditionalOStream>(
      std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  // Initiating the internal TimerOutput instance.
  if (external_timer.get() != nullptr)
    timer_output  = external_timer;
  else
    timer_output  = std::make_shared<dealii::TimerOutput>(
                      *pcout,
                      dealii::TimerOutput::summary,
                      dealii::TimerOutput::wall_times);

  // Initialize the quadrature formula
  const dealii::QGauss<dim>       quadrature_formula(3);

  const dealii::QGauss<dim-1>     face_quadrature_formula(3);

  quadrature_collection.push_back(quadrature_formula);

  face_quadrature_collection.push_back(face_quadrature_formula);

  // Initialize logger
  nonlinear_solver_logger.declare_column("Iteration");
  nonlinear_solver_logger.declare_column("Krylov-Iterations");
  nonlinear_solver_logger.declare_column("L2-Norm(Newton update)");
  nonlinear_solver_logger.declare_column("L2-Norm(Residual)");
  nonlinear_solver_logger.set_scientific("L2-Norm(Newton update)", true);
  nonlinear_solver_logger.set_scientific("L2-Norm(Residual)", true);

  decohesion_logger.declare_column("time");
  decohesion_logger.declare_column("max_effective_opening_displacement");
  decohesion_logger.declare_column("effective_opening_displacement");
  decohesion_logger.declare_column("normal_opening_displacement");
  decohesion_logger.declare_column("tangential_opening_displacement");
  decohesion_logger.declare_column("effective_cohesive_traction");
  decohesion_logger.declare_column("damage_variable");
  decohesion_logger.set_scientific("time", true);
  decohesion_logger.set_scientific("max_effective_opening_displacement", true);
  decohesion_logger.set_scientific("effective_opening_displacement", true);
  decohesion_logger.set_scientific("normal_opening_displacement", true);
  decohesion_logger.set_scientific("tangential_opening_displacement", true);
  decohesion_logger.set_scientific("effective_cohesive_traction", true);
  decohesion_logger.set_scientific("damage_variable", true);

  // Initialize supply term shared pointer
  supply_term = nullptr;
}


template <int dim>
const dealii::LinearAlgebraTrilinos::MPI::Vector &
GradientCrystalPlasticitySolver<dim>::get_damage_at_grain_boundaries()
{
  // The right-hand side of the projection is updated
  assemble_projection_rhs();

  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;

  distributed_solution.reinit(projection_rhs);

  distributed_solution = 0.;

  // The solver's tolerances are passed to the SolverControl instance
  // used to initialize the solver
  dealii::SolverControl solver_control(
    5000,
    std::max(projection_rhs.l2_norm() * 1e-8, 1e-9));

  dealii::LinearAlgebraTrilinos::SolverCG solver(solver_control);

  // The preconditioner is instanciated and initialized
  dealii::LinearAlgebraTrilinos::MPI::PreconditionAMG preconditioner;

  preconditioner.initialize(projection_matrix);

  // try-catch scope for the solve() call
  try
  {
    solver.solve(projection_matrix,
                 distributed_solution,
                 projection_rhs,
                 preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in the solve method: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception in the solve method!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }

  // Hanging node constraints are distributed and the solution is passed
  // to the ghost vector
  projection_hanging_node_constraints.distribute(distributed_solution);

  damage_variable_values = distributed_solution;

  return (damage_variable_values);
}




} // namespace gCP



// Explicit instantiations
template class gCP::GradientCrystalPlasticitySolver<2>;
template class gCP::GradientCrystalPlasticitySolver<3>;
