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
  const std::shared_ptr<dealii::TimerOutput>        external_timer)
:
parameters(parameters),
discrete_time(discrete_time),
fe_field(fe_field),
crystals_data(crystals_data),
elastic_strain(
  std::make_shared<Kinematics::ElasticStrain<dim>>(
    crystals_data,
    parameters.dimensionless_form_parameters.
      dimensionless_numbers[0])),
hooke_law(
  std::make_shared<ConstitutiveLaws::HookeLaw<dim>>(
    crystals_data,
    parameters.constitutive_laws_parameters.hooke_law_parameters,
    parameters.dimensionless_form_parameters.
      characteristic_quantities.stiffness)),
resolved_shear_stress_law(
  std::make_shared<ConstitutiveLaws::ResolvedShearStressLaw<dim>>(
    crystals_data)),
scalar_microstress_law(
  std::make_shared<ConstitutiveLaws::ScalarMicrostressLaw<dim>>(
    crystals_data,
    parameters.constitutive_laws_parameters.scalar_microstress_law_parameters,
    parameters.constitutive_laws_parameters.hardening_law_parameters,
    parameters.dimensionless_form_parameters.
      characteristic_quantities.slip_resistance)),
vectorial_microstress_law(
  std::make_shared<ConstitutiveLaws::VectorialMicrostressLaw<dim>>(
    crystals_data,
    parameters.constitutive_laws_parameters.vectorial_microstress_law_parameters)),
microtraction_law(
  std::make_shared<ConstitutiveLaws::MicrotractionLaw<dim>>(
    crystals_data,
    parameters.constitutive_laws_parameters.microtraction_law_parameters)),
cohesive_law(
  std::make_shared<ConstitutiveLaws::CohesiveLaw<dim>>(
    parameters.constitutive_laws_parameters.cohesive_law_parameters)),
degradation_function(
  std::make_shared<ConstitutiveLaws::DegradationFunction>(
    parameters.constitutive_laws_parameters.degradation_function_parameters)),
contact_law(
  std::make_shared<ConstitutiveLaws::ContactLaw<dim>>(
    parameters.constitutive_laws_parameters.contact_law_parameters,
    parameters.dimensionless_form_parameters.characteristic_quantities.stress,
    parameters.dimensionless_form_parameters.characteristic_quantities.displacement)),
//line_search(parameters.line_search_parameters),
nonlinear_solver_logger(
  parameters.logger_output_directory + "nonlinear_solver_log.txt"),
postprocessor(
  fe_field,
  crystals_data,
  parameters.dimensionless_form_parameters,
  true,
  true),
flag_init_was_called(false)
{
  Assert(fe_field.get() != nullptr,
         dealii::ExcMessage("The FEField<dim>'s shared pointer has "
                            "contains a nullptr."));
  Assert(crystals_data.get() != nullptr,
         dealii::ExcMessage("The CrystalsData<dim>'s shared pointer "
                            "contains a nullptr."));

  const bool &flag_dimensionless_formulation =
    parameters.dimensionless_form_parameters.
      flag_solve_dimensionless_problem;

  const bool &flag_microtraction_boundary_conditions =
    parameters.boundary_conditions_at_grain_boundaries ==
      RunTimeParameters::BoundaryConditionsAtGrainBoundaries::
        Microtraction;

  const bool &flag_rate_independent =
    parameters.constitutive_laws_parameters.
      scalar_microstress_law_parameters.flag_rate_independent;

  const bool flag_decohesion = parameters.allow_decohesion;

  if (flag_dimensionless_formulation && (flag_decohesion ||
        flag_microtraction_boundary_conditions))
  {
    Assert(false, dealii::ExcMessage(
      "The dimensionless formulation has not been implemented for the "
      "polycrystalline case of grain boundaries enhanced by a "
      "constitutive boundary condition and a cohesive law"));
  }

  if (flag_rate_independent && flag_microtraction_boundary_conditions)
  {
    Assert(false, dealii::ExcMessage(
      "The rate-independent formulation has not been implemented for "
      "the case of grain boundaries enhanced by a constitutive "
      "boundary condition"));
  }

  // Set macroscopic strain to zero
  macroscopic_strain = 0.;

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

  using Format = Utilities::Logger::Format;

  // Initialize logger
  nonlinear_solver_logger.declare_column("N-Itr", Format::Integer);
  nonlinear_solver_logger.declare_column("K-Itr", Format::Integer);
  nonlinear_solver_logger.declare_column("L-Itr", Format::Integer);
  nonlinear_solver_logger.declare_column("(NS)_L2", Format::Scientific);
  nonlinear_solver_logger.declare_column("(NS_U)_L2", Format::Scientific);
  nonlinear_solver_logger.declare_column("(NS_G)_L2", Format::Scientific);
  nonlinear_solver_logger.declare_column("(R)_L2", Format::Scientific);
  nonlinear_solver_logger.declare_column("(R_U)_L2", Format::Scientific);
  nonlinear_solver_logger.declare_column("(R_G)_L2", Format::Scientific);
  nonlinear_solver_logger.declare_column("C-Rate", Format::Decimal);

  /*!
   * @todo Debug
   */
  {
    table_handler.declare_column("LoadStep");
    table_handler.declare_column("Iterations");
    table_handler.declare_column("MonoAverageConvergence");
    table_handler.declare_column("MacroAverageConvergence");
    table_handler.declare_column("ReducedMacroAverageConvergence");
    table_handler.declare_column("MicroAverageConvergence");
    table_handler.set_precision("MonoAverageConvergence", 2);
    table_handler.set_precision("MacroAverageConvergence", 2);
    table_handler.set_precision("ReducedMacroAverageConvergence", 2);
    table_handler.set_precision("MicroAverageConvergence", 2);
  }

  postprocessor.init(hooke_law);

  // Initialize supply term shared pointer
  supply_term = nullptr;
}


template <int dim>
GradientCrystalPlasticitySolver<dim>::~GradientCrystalPlasticitySolver()
{
  /*line_search->write_to_file(
    parameters.logger_output_directory + "line_search_log.txt");*/
}


template <int dim>
const dealii::LinearAlgebraTrilinos::MPI::Vector &
GradientCrystalPlasticitySolver<dim>::get_damage_at_grain_boundaries()
{
  dealii::TimerOutput::Scope  t(*timer_output,
                                "Solver: Damage L2-Projection");

  // The right-hand side of the projection is updated
  assemble_projection_rhs();

  dealii::IndexSet locally_owned_dofs =
    projection_dof_handler.locally_owned_dofs();

  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_vector;

  distributed_vector.reinit(projection_rhs);

  distributed_vector = 0.0;

  for (unsigned int i = 0; i < lumped_projection_matrix.size(); ++i)
    if (locally_owned_dofs.is_element(i))
    {
      if (lumped_projection_matrix[i] != 0.0)
        distributed_vector[i] = projection_rhs[i] /
                                lumped_projection_matrix[i];
    }

  distributed_vector.compress(dealii::VectorOperation::insert);

  projection_hanging_node_constraints.distribute(distributed_vector);

  damage_variable_values = distributed_vector;

  return (damage_variable_values);
}



template <int dim>
void
GradientCrystalPlasticitySolver<dim>::output_data_to_file(
  std::ostream &file) const
{
  table_handler.write_text(
    file,
    dealii::TableHandler::TextOutputFormat::org_mode_table);
}



} // namespace gCP



// Explicit instantiations
template class gCP::GradientCrystalPlasticitySolver<2>;
template class gCP::GradientCrystalPlasticitySolver<3>;
