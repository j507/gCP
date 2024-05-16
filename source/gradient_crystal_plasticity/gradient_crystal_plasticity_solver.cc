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
    crystals_data)),
hooke_law(
  std::make_shared<ConstitutiveLaws::HookeLaw<dim>>(
    crystals_data,
    parameters.constitutive_laws_parameters.hooke_law_parameters)),
resolved_shear_stress_law(
  std::make_shared<ConstitutiveLaws::ResolvedShearStressLaw<dim>>(
    crystals_data)),
scalar_microstress_law(
  std::make_shared<ConstitutiveLaws::ScalarMicrostressLaw<dim>>(
    crystals_data,
    parameters.constitutive_laws_parameters.scalar_microstress_law_parameters,
    parameters.constitutive_laws_parameters.hardening_law_parameters)),
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
    parameters.constitutive_laws_parameters.contact_law_parameters)),
residual_norm(std::numeric_limits<double>::max()),
line_search(parameters.line_search_parameters),
nonlinear_solver_logger(
  parameters.logger_output_directory + "nonlinear_solver_log.txt"),
print_out(true),
postprocessor(
  fe_field,
  crystals_data),
flag_init_was_called(false),
trial_microstress(
  fe_field->get_triangulation(),
  fe_field->get_slips_fe_degree())
{
  Assert(fe_field.get() != nullptr,
         dealii::ExcMessage("The FEField<dim>'s shared pointer has "
                            "contains a nullptr."));
  Assert(crystals_data.get() != nullptr,
         dealii::ExcMessage("The CrystalsData<dim>'s shared pointer "
                            "contains a nullptr."));

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

  // Initialize logger
  nonlinear_solver_logger.declare_column("N-Itr");
  nonlinear_solver_logger.declare_column("K-Itr");
  nonlinear_solver_logger.declare_column("L-Itr");
  nonlinear_solver_logger.declare_column("(NS)_L2");
  nonlinear_solver_logger.declare_column("(NS_U)_L2");
  nonlinear_solver_logger.declare_column("(NS_G)_L2");
  nonlinear_solver_logger.declare_column("(R)_L2");
  nonlinear_solver_logger.declare_column("(R_U)_L2");
  nonlinear_solver_logger.declare_column("(R_G)_L2");
  nonlinear_solver_logger.declare_column("C-Rate");
  nonlinear_solver_logger.set_scientific("(NS)_L2", true);
  nonlinear_solver_logger.set_scientific("(NS_U)_L2", true);
  nonlinear_solver_logger.set_scientific("(NS_G)_L2", true);
  nonlinear_solver_logger.set_scientific("(R)_L2", true);
  nonlinear_solver_logger.set_scientific("(R_U)_L2", true);
  nonlinear_solver_logger.set_scientific("(R_G)_L2", true);

  /*!
   * @brief Code snippet only to be considered by bi-crystal simulations
   *
   * @todo Docu
   */
  {
    table_handler.declare_column("effective_opening_displacement");
    table_handler.declare_column("effective_traction_vector");
    table_handler.declare_column("time");
    table_handler.declare_column("damage_variable");
    table_handler.set_scientific("effective_opening_displacement", true);
    table_handler.set_scientific("effective_traction_vector", true);
    table_handler.set_scientific("time", true);
    table_handler.set_scientific("damage_variable", true);
    table_handler.set_precision("effective_opening_displacement", 6);
    table_handler.set_precision("effective_traction_vector", 6);
    table_handler.set_precision("time", 6);
    table_handler.set_precision("damage_variable", 6);
  }

  postprocessor.init(hooke_law);

  // Initialize supply term shared pointer
  supply_term = nullptr;
}


template <int dim>
GradientCrystalPlasticitySolver<dim>::~GradientCrystalPlasticitySolver()
{
  line_search.write_to_file(
    parameters.logger_output_directory + "line_search_log.txt");
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
