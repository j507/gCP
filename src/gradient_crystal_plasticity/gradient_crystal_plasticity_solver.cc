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
residual_norm(std::numeric_limits<double>::max()),
nonlinear_solver_logger(parameters.logger_output_directory +
                        "nonlinear_solver_log.txt"),
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
  nonlinear_solver_logger.declare_column("L2-Norm(Newton update)");
  nonlinear_solver_logger.declare_column("L2-Norm(Residual)");
  nonlinear_solver_logger.set_scientific("L2-Norm(Newton update)", true);
  nonlinear_solver_logger.set_scientific("L2-Norm(Residual)", true);
}



} // namespace gCP



// Explicit instantiations
template class gCP::GradientCrystalPlasticitySolver<2>;
template class gCP::GradientCrystalPlasticitySolver<3>;
