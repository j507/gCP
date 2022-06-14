#include <gCP/postprocessing.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/filtered_iterator.h>

namespace gCP
{



namespace Postprocessing
{



template <int dim>
SimpleShear<dim>::SimpleShear(
  std::shared_ptr<FEField<dim>>         &fe_field,
  std::shared_ptr<dealii::Mapping<dim>> &mapping,
  const double                          shear_at_upper_boundary,
  const dealii::types::boundary_id      upper_boundary_id,
  const double                          width)
:
fe_field(fe_field),
mapping_collection(*mapping),
dof_handler(fe_field->get_triangulation()),
projection_rhs(2),
projected_data(2),
shear_at_upper_boundary(shear_at_upper_boundary),
upper_boundary_id(upper_boundary_id),
width(width),
flag_init_was_called(false)
{
  fe_collection.push_back(dealii::FE_Q<dim>(1));

  const dealii::QGauss<dim> quadrature_formula(3);

  quadrature_collection.push_back(quadrature_formula);

  // Setting up columns
  table_handler.declare_column("shear at upper boundary");
  table_handler.declare_column("stress 12 at upper boundary");

  // Setting all columns to scientific notation
  table_handler.set_scientific("shear at upper boundary", true);
  table_handler.set_scientific("stress 12 at upper boundary", true);

  // Setting columns' precision
  table_handler.set_precision("shear at upper boundary", 6);
  table_handler.set_precision("stress 12 at upper boundary", 6);
}



template <int dim>
void SimpleShear<dim>::init(
  std::shared_ptr<const Kinematics::ElasticStrain<dim>>   elastic_strain,
  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law)
{
  this->elastic_strain  = elastic_strain;
  this->hooke_law       = hooke_law;

  dof_handler.distribute_dofs(fe_collection);

  dealii::DoFRenumbering::Cuthill_McKee(dof_handler);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                  locally_relevant_dofs);

  hanging_node_constraints.clear();
  {
    hanging_node_constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                    hanging_node_constraints);
  }
  hanging_node_constraints.close();

  // Initiate the matrix
  {
    dealii::TrilinosWrappers::SparsityPattern
      sparsity_pattern(locally_owned_dofs,
                       locally_owned_dofs,
                       locally_relevant_dofs,
                       MPI_COMM_WORLD);

    dealii::DoFTools::make_sparsity_pattern(
      dof_handler,
      sparsity_pattern,
      hanging_node_constraints,
      false,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

    sparsity_pattern.compress();

    projection_matrix.reinit(sparsity_pattern);
  }

  for (unsigned int i = 0; i < projected_data.size(); ++i)
  {
    projected_data[i].reinit(locally_relevant_dofs,
                             MPI_COMM_WORLD);
    projection_rhs[i].reinit(locally_owned_dofs,
                             locally_relevant_dofs,
                             MPI_COMM_WORLD,
                             true);
  }

  assemble_projection_matrix();

  flag_init_was_called = true;
}



template <int dim>
void SimpleShear<dim>::assemble_projection_matrix()
{
  // Set up local aliases
  using CellIterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;

  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  // Reset data
  projection_matrix = 0.0;

  // Set up the lambda function for the local assembly operation
  auto worker = [this](
    const CellIterator                                                &cell,
    gCP::AssemblyData::Postprocessing::ProjectionMatrix::Scratch<dim> &scratch,
    gCP::AssemblyData::Postprocessing::ProjectionMatrix::Copy         &data)
  {
    this->assemble_local_projection_matrix(cell, scratch, data);
  };

  // Set up the lambda function for the copy local to global operation
  auto copier = [this](
    const gCP::AssemblyData::Postprocessing::ProjectionMatrix::Copy &data)
  {
    this->copy_local_to_global_projection_matrix(data);
  };

  // Define the update flags for the FEValues instances
  const dealii::UpdateFlags update_flags =
    dealii::update_JxW_values |
    dealii::update_values;

  // Assemble using the WorkStream approach
  dealii::WorkStream::run(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.end()),
    worker,
    copier,
    gCP::AssemblyData::Postprocessing::ProjectionMatrix::Scratch<dim>(
      mapping_collection,
      quadrature_collection,
      fe_collection,
      update_flags),
    gCP::AssemblyData::Postprocessing::ProjectionMatrix::Copy(
      fe_collection.max_dofs_per_cell()));

  // Compress global data
  projection_matrix.compress(dealii::VectorOperation::add);
}


template <int dim>
void SimpleShear<dim>::assemble_local_projection_matrix(
  const typename dealii::DoFHandler<dim>::active_cell_iterator      &cell,
  gCP::AssemblyData::Postprocessing::ProjectionMatrix::Scratch<dim> &scratch,
  gCP::AssemblyData::Postprocessing::ProjectionMatrix::Copy         &data)
{
  // Reset local data
  data.local_matrix = 0.0;

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  const dealii::FEValuesExtractors::Scalar  extractor(0);

  // Update the hp::FEValues instance to the values of the current cell
  scratch.hp_fe_values.reinit(cell);

  const dealii::FEValues<dim> &fe_values =
    scratch.hp_fe_values.get_present_fe_values();

  // Get JxW values at the quadrature points
  scratch.JxW_values = fe_values.get_JxW_values();

  // Loop over quadrature points
  for (unsigned int q_point = 0; q_point < scratch.n_q_points; ++q_point)
  {
    // Extract test function values at the quadrature points (Displacement)
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.scalar_phi[i] = fe_values[extractor].value(i,q_point);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
            data.local_matrix(i,j) +=
              scratch.scalar_phi[i] *
              scratch.scalar_phi[j] *
              scratch.JxW_values[q_point];
  } // Loop over quadrature points
}



template <int dim>
void SimpleShear<dim>::copy_local_to_global_projection_matrix(
  const gCP::AssemblyData::Postprocessing::ProjectionMatrix::Copy &data)
{
  hanging_node_constraints.distribute_local_to_global(
    data.local_matrix,
    data.local_dof_indices,
    projection_matrix);
}



template <int dim>
void SimpleShear<dim>::compute_data(const double time)
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The SimpleShear<dim>"
                                  " instance has not been"
                                  " initialized."));

  compute_stress_12_at_boundary();

  compute_strain_12();

  table_handler.add_value("shear at upper boundary", time * shear_at_upper_boundary);
  table_handler.add_value("stress 12 at upper boundary", average_stress_12);
}



template <int dim>
void SimpleShear<dim>::compute_stress_12_at_boundary()
{
  // Initiate the local integral value and at each wall.
  average_stress_12 = 0.0;

  dealii::hp::QCollection<dim-1>  face_quadrature_collection;
  {
    const dealii::QGauss<dim-1>     face_quadrature_formula(3);

    face_quadrature_collection.push_back(face_quadrature_formula);
  }

  const dealii::UpdateFlags face_update_flags =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_gradients;

  // Finite element values
  dealii::hp::FEFaceValues<dim> hp_fe_face_values(
    mapping_collection,
    fe_field->get_fe_collection(),
    face_quadrature_collection,
    face_update_flags);

  // Number of quadrature points
  const unsigned int n_face_q_points =
    face_quadrature_collection.max_n_quadrature_points();

  // Vectors to stores the temperature gradients and normal vectors
  // at the quadrature points
  std::vector<double>                           JxW_values(n_face_q_points);
  std::vector<dealii::SymmetricTensor<2, dim>>  strain_tensor_values(n_face_q_points);
  std::vector<dealii::SymmetricTensor<2, dim>>  elastic_strain_tensor_values(n_face_q_points);
  std::vector<dealii::SymmetricTensor<2, dim>>  stress_tensor_values(n_face_q_points);
  std::vector<std::vector<double>>              slip_values(fe_field->get_n_slips(),
                                                            std::vector<double>(n_face_q_points));

  double stress_12        = 0.0;
  double local_stress_12  = 0.0;

  for (const auto &cell : fe_field->get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && face->boundary_id() == upper_boundary_id)
        {
          // Reset local face integral values
          local_stress_12 = 0.0;

          // Get the crystal identifier for the current cell
          const unsigned int crystal_id = cell->active_fe_index();

          // Update the hp::FEFaceValues instance to the values of the current cell
          hp_fe_face_values.reinit(cell, face);

          const dealii::FEFaceValues<dim> &fe_face_values =
            hp_fe_face_values.get_present_fe_values();

          // Get JxW values at the quadrature points
          JxW_values = fe_face_values.get_JxW_values();

          // Get the displacement gradients at the quadrature points
          fe_face_values[fe_field->get_displacement_extractor(crystal_id)].get_function_symmetric_gradients(
            fe_field->solution,
            strain_tensor_values);

          // Get the slips and their gradients values at the quadrature points
          for (unsigned int slip_id = 0;
              slip_id < fe_field->get_n_slips();
              ++slip_id)
          {
            fe_face_values[fe_field->get_slip_extractor(crystal_id, slip_id)].get_function_values(
              fe_field->solution,
              slip_values[slip_id]);
          }

          // Numerical integration
          for (unsigned int face_q_point = 0;
               face_q_point < n_face_q_points;
               ++face_q_point)
          {
            // Compute the elastic strain tensor at the quadrature point
            elastic_strain_tensor_values[face_q_point] =
              elastic_strain->get_elastic_strain_tensor(
                crystal_id,
                face_q_point,
                strain_tensor_values[face_q_point],
                slip_values);

            // Compute the stress tensor at the quadrature point
            stress_tensor_values[face_q_point] =
              hooke_law->get_stress_tensor(
                crystal_id,
                elastic_strain_tensor_values[face_q_point]);

            local_stress_12 +=
              stress_tensor_values[face_q_point][0][1] *
              JxW_values[face_q_point];
          }

          stress_12 += local_stress_12;
        }

  // Gather the values of each processor
  stress_12 = dealii::Utilities::MPI::sum(stress_12, MPI_COMM_WORLD);

  average_stress_12 = stress_12 / width;
}



template <int dim>
void SimpleShear<dim>::compute_strain_12()
{
  assemble_projection_rhs();

  project();
}



template <int dim>
void SimpleShear<dim>::assemble_projection_rhs()
{
  // Set up local aliases
  using CellIterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;

  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  // Reset data
  for (auto &right_hand_side: projection_rhs)
    right_hand_side = 0.0;

  // Set up the lambda function for the local assembly operation
  auto worker = [this](
    const CellIterator                                             &cell,
    gCP::AssemblyData::Postprocessing::ProjectionRHS::Scratch<dim> &scratch,
    gCP::AssemblyData::Postprocessing::ProjectionRHS::Copy         &data)
  {
    this->assemble_local_projection_rhs(cell, scratch, data);
  };

  // Set up the lambda function for the copy local to global operation
  auto copier = [this](
    const gCP::AssemblyData::Postprocessing::ProjectionRHS::Copy &data)
  {
    this->copy_local_to_global_projection_rhs(data);
  };

  // Define the update flags for the FEValues instances
  const dealii::UpdateFlags scalar_update_flags =
    dealii::update_JxW_values |
    dealii::update_values;

  const dealii::UpdateFlags vector_update_flags =
    dealii::update_values |
    dealii::update_gradients;

  // Assemble using the WorkStream approach
  dealii::WorkStream::run(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.end()),
    worker,
    copier,
    gCP::AssemblyData::Postprocessing::ProjectionRHS::Scratch<dim>(
      mapping_collection,
      quadrature_collection,
      fe_collection,
      scalar_update_flags,
      fe_field->get_fe_collection(),
      vector_update_flags,
      fe_field->get_n_slips()),
    gCP::AssemblyData::Postprocessing::ProjectionRHS::Copy(
      fe_collection.max_dofs_per_cell()));

  // Compress global data
  for (auto &right_hand_side: projection_rhs)
    right_hand_side.compress(dealii::VectorOperation::add);
}


template <int dim>
void SimpleShear<dim>::assemble_local_projection_rhs(
  const typename dealii::DoFHandler<dim>::active_cell_iterator    &cell,
  gCP::AssemblyData::Postprocessing::ProjectionRHS::Scratch<dim>  &scratch,
  gCP::AssemblyData::Postprocessing::ProjectionRHS::Copy          &data)
{
  // Reset local data
  for (auto &local_right_hand_side: data.local_rhs)
    local_right_hand_side = 0.0;

  data.local_matrix_for_inhomogeneous_bcs = 0.0;

  // Local to global mapping of the indices of the degrees of freedom
  cell->get_dof_indices(data.local_dof_indices);

  const dealii::FEValuesExtractors::Scalar  extractor(0);

  // Get the crystal identifier for the current cell
  const unsigned int crystal_id = cell->active_fe_index();

  // Update the hp::FEValues instance to the values of the current cell
  scratch.scalar_hp_fe_values.reinit(cell);

  const dealii::FEValues<dim> &scalar_fe_values =
    scratch.scalar_hp_fe_values.get_present_fe_values();

  typename dealii::DoFHandler<dim>::active_cell_iterator
    vector_cell(&fe_field->get_triangulation(),
                cell->level(),
                cell->index(),
                &fe_field->get_dof_handler());

  scratch.vector_hp_fe_values.reinit(vector_cell);

  const dealii::FEValues<dim> &vector_fe_values =
    scratch.vector_hp_fe_values.get_present_fe_values();

  // Get JxW values at the quadrature points
  scratch.JxW_values = scalar_fe_values.get_JxW_values();

  // Get the linear strain tensor values at the quadrature points
  vector_fe_values[fe_field->get_displacement_extractor(crystal_id)].get_function_symmetric_gradients(
    fe_field->solution,
    scratch.strain_tensor_values);

  // Get the slips and their gradients values at the quadrature points
  for (unsigned int slip_id = 0; slip_id < scratch.n_slips; ++slip_id)
  {
    vector_fe_values[fe_field->get_slip_extractor(crystal_id, slip_id)].get_function_values(
      fe_field->solution,
      scratch.slip_values[slip_id]);
  }

  // Loop over quadrature points
  for (unsigned int q_point = 0; q_point < scratch.n_q_points; ++q_point)
  {
    // Compute the elastic strain tensor at the quadrature point
    scratch.elastic_strain_tensor_values[q_point] =
      elastic_strain->get_elastic_strain_tensor(
        crystal_id,
        q_point,
        scratch.strain_tensor_values[q_point],
        scratch.slip_values);

    // Compute the stress tensor at the quadrature point
    scratch.stress_tensor_values[q_point] =
      hooke_law->get_stress_tensor(
        crystal_id,
        scratch.elastic_strain_tensor_values[q_point]);

    // Extract test function values at the quadrature points (Slips)
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      scratch.scalar_phi[i] =
        scalar_fe_values[extractor].value(i,q_point);

    // Loop over the degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs[0](i) +=
        scratch.scalar_phi[i] *
        2.0 * scratch.strain_tensor_values[q_point][0][1] *
        scratch.JxW_values[q_point];

      data.local_rhs[1](i) +=
        scratch.scalar_phi[i] *
        scratch.stress_tensor_values[q_point][0][1] *
        scratch.JxW_values[q_point];
    }
  } // Loop over quadrature points
}

template <int dim>
void SimpleShear<dim>::copy_local_to_global_projection_rhs(
  const gCP::AssemblyData::Postprocessing::ProjectionRHS::Copy &data)
{
  for (unsigned int i = 0; i < data.local_rhs.size(); ++i)
    hanging_node_constraints.distribute_local_to_global(
      data.local_rhs[i],
      data.local_dof_indices,
      projection_rhs[i],
      data.local_matrix_for_inhomogeneous_bcs);
}



template <int dim>
void SimpleShear<dim>::project()
{
// In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;

  distributed_solution.reinit(projection_rhs[0]);

  distributed_solution = 0.;

  for (unsigned int i = 0; i < projection_rhs.size(); ++i)
  {
    // The solver's tolerances are passed to the SolverControl instance
    // used to initialize the solver
    dealii::SolverControl solver_control(
      5000,
      std::max(projection_rhs[i].l2_norm() * 1e-8, 1e-9));

    dealii::LinearAlgebraTrilinos::SolverCG solver(solver_control);

    // try-catch scope for the solve() call
    try
    {
      solver.solve(projection_matrix,
                  distributed_solution,
                  projection_rhs[i],
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

    hanging_node_constraints.distribute(
      distributed_solution);

    projected_data[i]  = distributed_solution;
  }
}


template <int dim>
void SimpleShear<dim>::output_data_to_file(
  std::ostream &file) const
{
  table_handler.write_text(
    file,
    dealii::TableHandler::TextOutputFormat::org_mode_table);
}



} // namespace Postprocessing



} // namespace gCP



template class gCP::Postprocessing::SimpleShear<2>;
template class gCP::Postprocessing::SimpleShear<3>;
