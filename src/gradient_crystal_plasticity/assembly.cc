#include <gCP/assembly_data.h>
#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace gCP
{



template <int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_jacobian()
{
  if (parameters.verbose)
    *pcout << std::setw(38) << std::left
           << "  Solver: Assembling jacobian...";

  dealii::TimerOutput::Scope  t(*timer_output,
                                "Solver: Jacobian assembly");

  // Set up local aliases
  using CellIterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;

  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  // Reset data
  jacobian = 0.0;

  // Set up the lambda function for the local assembly operation
  auto worker = [this](
    const CellIterator                         &cell,
    gCP::AssemblyData::Jacobian::Scratch<dim>  &scratch,
    gCP::AssemblyData::Jacobian::Copy          &data)
  {
    this->assemble_local_jacobian(cell, scratch, data);
  };

  // Set up the lambda function for the copy local to global operation
  auto copier = [this](const gCP::AssemblyData::Jacobian::Copy &data)
  {
    this->copy_local_to_global_jacobian(data);
  };

  // Define the update flags for the FEValues instances
  const dealii::UpdateFlags update_flags =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_gradients |
    dealii::update_quadrature_points;

  //dealii::UpdateFlags face_update_flags = dealii::update_JxW_values;

  // Assemble using the WorkStream approach
  dealii::WorkStream::run(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().end()),
    worker,
    copier,
    gCP::AssemblyData::Jacobian::Scratch<dim>(
      mapping_collection,
      quadrature_collection,
      fe_field->get_fe_collection(),
      update_flags,
      crystals_data->get_n_slips()),
    gCP::AssemblyData::Jacobian::Copy(
      fe_field->get_fe_collection().max_dofs_per_cell()));

  // Compress global data
  jacobian.compress(dealii::VectorOperation::add);

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_local_jacobian(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  gCP::AssemblyData::Jacobian::Scratch<dim>                     &scratch,
  gCP::AssemblyData::Jacobian::Copy                             &data)
{
  // Reset local data
  data.local_matrix = 0.0;

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Get the crystal identifier for the current cell
  const unsigned int crystal_id = cell->active_fe_index();

  // Get the stiffness tetrad of the current crystal
  scratch.stiffness_tetrad =
    hooke_law->get_stiffness_tetrad(crystal_id);

  // Get the slips' reduced gradient hardening tensors of the current
  // crystal
  scratch.reduced_gradient_hardening_tensors =
    vector_microscopic_stress_law->get_reduced_gradient_hardening_tensors(crystal_id);

  // Get the slips' symmetrized Schmid tensors of the current crystal
  scratch.symmetrized_schmid_tensors =
    crystals_data->get_symmetrized_schmid_tensors(crystal_id);

  // Update the hp::FEValues instance to the values of the current cell
  scratch.hp_fe_values.reinit(cell);

  const dealii::FEValues<dim> &fe_values =
    scratch.hp_fe_values.get_present_fe_values();

  // Get JxW values at the quadrature points
  scratch.JxW_values = fe_values.get_JxW_values();

  // Get values of the hardening variable at the quadrature points
  const std::vector<std::shared_ptr<QuadraturePointHistory<dim>>>
    local_quadrature_point_history =
      quadrature_point_history.get_data(cell);

  // Get values of the slips at the quadrature points
  for (unsigned int slip_id = 0;
      slip_id < crystals_data->get_n_slips();
      ++slip_id)
  {
    fe_values[fe_field->get_slip_extractor(crystal_id, slip_id)].get_function_values(
      solution,
      scratch.slip_values[slip_id]);

    fe_values[fe_field->get_slip_extractor(crystal_id, slip_id)].get_function_values(
      fe_field->old_solution,
      scratch.old_slip_values[slip_id]);
  }

  // Loop over quadrature points
  for (unsigned int q_point = 0; q_point < scratch.n_q_points; ++q_point)
  {
    // Compute the Gateaux derivative values of the scalar microscopic
    // stress w.r.t. slip
    scratch.gateaux_derivative_values[q_point] =
      scalar_microscopic_stress_law->get_gateaux_derivative_matrix(
        q_point,
        scratch.slip_values,
        scratch.old_slip_values,
        local_quadrature_point_history[q_point],
        discrete_time.get_next_step_size());

    // Extract test function values at the quadrature points (Displacement)
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.sym_grad_vector_phi[i] =
        fe_values[fe_field->get_displacement_extractor(crystal_id)].symmetric_gradient(i,q_point);
    }

    // Extract test function values at the quadrature points (Slips)
    for (unsigned int slip_id = 0;
         slip_id < crystals_data->get_n_slips();
         ++slip_id)
    {
      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      {
        scratch.scalar_phi[slip_id][i] =
          fe_values[fe_field->get_slip_extractor(crystal_id, slip_id)].value(i,q_point);

        scratch.grad_scalar_phi[slip_id][i] =
          fe_values[fe_field->get_slip_extractor(crystal_id, slip_id)].gradient(i,q_point);
      }
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
      {
        if (fe_field->get_global_component(crystal_id, i) < dim)
        {
          if (fe_field->get_global_component(crystal_id, j) < dim)
          {
            data.local_matrix(i,j) +=
              scratch.sym_grad_vector_phi[i] *
              scratch.stiffness_tetrad *
              scratch.sym_grad_vector_phi[j] *
              scratch.JxW_values[q_point];
          }
          else
          {
            const unsigned int slip_id_beta =
              fe_field->get_global_component(crystal_id, j) - dim;

            data.local_matrix(i,j) -=
              scratch.sym_grad_vector_phi[i] *
              scratch.stiffness_tetrad *
              scratch.symmetrized_schmid_tensors[slip_id_beta] *
              scratch.scalar_phi[slip_id_beta][j] *
              scratch.JxW_values[q_point];
          }
        }
        else
        {
          const unsigned int slip_id_alpha =
              fe_field->get_global_component(crystal_id, i) - dim;

          if (fe_field->get_global_component(crystal_id, j) < dim)
          {
            data.local_matrix(i,j) -=
              scratch.scalar_phi[slip_id_alpha][i] *
              scratch.symmetrized_schmid_tensors[slip_id_alpha] *
              scratch.stiffness_tetrad *
              scratch.sym_grad_vector_phi[j] *
              scratch.JxW_values[q_point];
          }
          else
          {
            const unsigned int slip_id_beta =
                fe_field->get_global_component(crystal_id, j) - dim;

            data.local_matrix(i,j) +=
              (((slip_id_alpha == slip_id_beta) ?
                scratch.grad_scalar_phi[slip_id_alpha][i] *
                scratch.reduced_gradient_hardening_tensors[slip_id_alpha] *
                scratch.grad_scalar_phi[slip_id_beta][j]
                : 0.0)
               -
               scratch.scalar_phi[slip_id_alpha][i] *
               (-1.0 *
                scratch.symmetrized_schmid_tensors[slip_id_alpha] *
                scratch.stiffness_tetrad *
                scratch.symmetrized_schmid_tensors[slip_id_beta]
                -
                scratch.gateaux_derivative_values[q_point][slip_id_alpha][slip_id_beta]) *
               scratch.scalar_phi[slip_id_beta][j])*
              scratch.JxW_values[q_point];
          }
        }
      } // Loop over local degrees of freedom
  } // Loop over quadrature points
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::copy_local_to_global_jacobian(
  const gCP::AssemblyData::Jacobian::Copy &data)
{
  fe_field->get_newton_method_constraints().distribute_local_to_global(
    data.local_matrix,
    data.local_dof_indices,
    jacobian);
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_residual()
{
  if (parameters.verbose)
    *pcout << std::setw(38) << std::left
           << "  Solver: Assembling residual...";

  dealii::TimerOutput::Scope  t(*timer_output,
                                "Solver: Residual assembly");

  // Set up local aliases
  using CellIterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;

  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  // Reset data
  residual = 0.0;

  // Set up the lambda function for the local assembly operation
  auto worker = [this](
    const CellIterator                         &cell,
    gCP::AssemblyData::Residual::Scratch<dim>  &scratch,
    gCP::AssemblyData::Residual::Copy          &data)
  {
    this->assemble_local_residual(cell, scratch, data);
  };

  // Set up the lambda function for the copy local to global operation
  auto copier = [this](const gCP::AssemblyData::Residual::Copy  &data)
  {
    this->copy_local_to_global_residual(data);
  };

  // Define the update flags for the FEValues instances
  const dealii::UpdateFlags update_flags  =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_gradients |
    dealii::update_quadrature_points;

  const dealii::UpdateFlags face_update_flags  =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_quadrature_points;

  // Assemble using the WorkStream approach
  dealii::WorkStream::run(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().end()),
    worker,
    copier,
    gCP::AssemblyData::Residual::Scratch<dim>(
      mapping_collection,
      quadrature_collection,
      face_quadrature_collection,
      fe_field->get_fe_collection(),
      update_flags,
      face_update_flags,
      crystals_data->get_n_slips()),
    gCP::AssemblyData::Residual::Copy(
      fe_field->get_fe_collection().max_dofs_per_cell()));

  // Compress global data
  residual.compress(dealii::VectorOperation::add);

  residual_norm = residual.l2_norm();

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_local_residual(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  gCP::AssemblyData::Residual::Scratch<dim>                     &scratch,
  gCP::AssemblyData::Residual::Copy                             &data)
{
  // Reset local data
  data.local_rhs                          = 0.0;
  data.local_matrix_for_inhomogeneous_bcs = 0.0;

  // Local to global mapping of the indices of the degrees of freedom
  cell->get_dof_indices(data.local_dof_indices);

  // Get the crystal identifier for the current cell
  const unsigned int crystal_id = cell->active_fe_index();

  // Update the hp::FEValues instance to the values of the current cell
  scratch.hp_fe_values.reinit(cell);

  const dealii::FEValues<dim> &fe_values =
    scratch.hp_fe_values.get_present_fe_values();

  // Get JxW values at the quadrature points
  scratch.JxW_values = fe_values.get_JxW_values();

  // Get the hardening variable values at the quadrature points
  const std::vector<std::shared_ptr<QuadraturePointHistory<dim>>>
    local_quadrature_point_history =
      quadrature_point_history.get_data(cell);

  // Get the linear strain tensor values at the quadrature points
  fe_values[fe_field->get_displacement_extractor(crystal_id)].get_function_symmetric_gradients(
    solution,
    scratch.strain_tensor_values);

  // Get the supply term values at the quadrature points
  supply_term->value_list(
    fe_values.get_quadrature_points(),
    scratch.supply_term_values);

  // Get the slips and their gradients values at the quadrature points
  for (unsigned int slip_id = 0;
      slip_id < crystals_data->get_n_slips();
      ++slip_id)
  {
    fe_values[fe_field->get_slip_extractor(crystal_id, slip_id)].get_function_values(
      solution,
      scratch.slip_values[slip_id]);

    fe_values[fe_field->get_slip_extractor(crystal_id, slip_id)].get_function_values(
      fe_field->old_solution,
      scratch.old_slip_values[slip_id]);

    fe_values[fe_field->get_slip_extractor(crystal_id, slip_id)].get_function_gradients(
      solution,
      scratch.slip_gradient_values[slip_id]);
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

    // Compute the resolved stress, scalar microscopic stress and
    // vector microscopic stress values at the quadrature point
    for (unsigned int slip_id = 0;
         slip_id < crystals_data->get_n_slips();
         ++slip_id)
    {
      scratch.vector_microscopic_stress_values[slip_id][q_point] =
        vector_microscopic_stress_law->get_vector_microscopic_stress(
          crystal_id,
          slip_id,
          scratch.slip_gradient_values[slip_id][q_point]);

      scratch.resolved_stress_values[slip_id][q_point] =
        resolved_shear_stress_law->get_resolved_shear_stress(
          crystal_id,
          slip_id,
          scratch.stress_tensor_values[q_point]);

      scratch.scalar_microscopic_stress_values[slip_id][q_point] =
        scalar_microscopic_stress_law->get_scalar_microscopic_stress(
          scratch.slip_values[slip_id][q_point],
          scratch.old_slip_values[slip_id][q_point],
          local_quadrature_point_history[q_point]->get_slip_resistance(slip_id),
          discrete_time.get_next_step_size());
    }

    // Extract test function values at the quadrature points (Displacements)
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.vector_phi[i] =
        fe_values[fe_field->get_displacement_extractor(crystal_id)].value(i,q_point);

      scratch.sym_grad_vector_phi[i] =
        fe_values[fe_field->get_displacement_extractor(crystal_id)].symmetric_gradient(i,q_point);
    }

    // Extract test function values at the quadrature points (Slips)
    for (unsigned int slip_id = 0;
         slip_id < crystals_data->get_n_slips();
         ++slip_id)
    {
      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      {
        scratch.scalar_phi[slip_id][i] =
          fe_values[fe_field->get_slip_extractor(crystal_id, slip_id)].value(i,q_point);

        scratch.grad_scalar_phi[slip_id][i] =
          fe_values[fe_field->get_slip_extractor(crystal_id, slip_id)].gradient(i,q_point);
      }
    }

    // Loop over the degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      if (fe_field->get_global_component(crystal_id, i) < dim)
      {
        data.local_rhs(i) -=
          (scratch.sym_grad_vector_phi[i] *
           scratch.stress_tensor_values[q_point]
           -
           scratch.vector_phi[i] *
           scratch.supply_term_values[q_point]) *
          scratch.JxW_values[q_point];
      }
      else
      {
        const unsigned int slip_id =
                fe_field->get_global_component(crystal_id, i) - dim;

        data.local_rhs(i) -=
          (scratch.grad_scalar_phi[slip_id][i] *
           scratch.vector_microscopic_stress_values[slip_id][q_point]
           -
           scratch.scalar_phi[slip_id][i] *
           (scratch.resolved_stress_values[slip_id][q_point]
            -
            scratch.scalar_microscopic_stress_values[slip_id][q_point])) *
          scratch.JxW_values[q_point];
      }
    } // Loop over the degrees of freedom
  } // Loop over quadrature points

  /*
  if (cell->at_boundary())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == 1)
      {
        // Update the hp::FEFaceValues instance to the values of the current cell
        scratch.hp_fe_face_values.reinit(cell, face);

        const dealii::FEFaceValues<dim> &fe_face_values =
          scratch.hp_fe_face_values.get_present_fe_values();

        // Compute the Neumann boundary function values at the
        // quadrature points
        neumann_boundary_function.value_list(
          fe_face_values.get_quadrature_points(),
          scratch.neumann_boundary_values);

        // Loop over face quadrature points
        for (unsigned int q_point = 0; q_point < scratch.n_face_q_points; ++q_point)
        {
          // Extract the test function's values at the face quadrature points
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            scratch.face_phi[i] =
              fe_face_values[fe_field->get_displacement_extractor(crystal_id)].value(i,q_point);

          // Mapped quadrature weight
          double da = fe_face_values.JxW(q_point);

          // Loop over degrees of freedom
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            data.local_rhs(i) -=
              scratch.face_phi[i] *
              scratch.neumann_boundary_values[q_point] *
              da;
        } // Loop over face quadrature points
      } // if (face->at_boundary() && face->boundary_id() == 3)
  */
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::copy_local_to_global_residual(
  const gCP::AssemblyData::Residual::Copy &data)
{
  fe_field->get_newton_method_constraints().distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    residual,
    data.local_matrix_for_inhomogeneous_bcs);
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::update_quadrature_point_history()
{
  dealii::TimerOutput::Scope  t(*timer_output,
                                "Solver: Update quadrature point history");

  // Set up local aliases
  using CellIterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;

  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  // Set up the lambda function for the local assembly operation
  auto worker = [this](
    const CellIterator                                      &cell,
    gCP::AssemblyData::QuadraturePointHistory::Scratch<dim> &scratch,
    gCP::AssemblyData::QuadraturePointHistory::Copy         &data)
  {
    this->update_local_quadrature_point_history(cell, scratch, data);
  };

  // Set up the lambda function for the copy local to global operation
  auto copier = [this](const gCP::AssemblyData::QuadraturePointHistory::Copy  &data)
  {
    this->copy_local_to_global_quadrature_point_history(data);
  };

  // Define the update flags for the FEValues instances
  const dealii::UpdateFlags update_flags  =
    dealii::update_values;

  // Assemble using the WorkStream approach
  dealii::WorkStream::run(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().end()),
    worker,
    copier,
    gCP::AssemblyData::QuadraturePointHistory::Scratch<dim>(
      mapping_collection,
      quadrature_collection,
      fe_field->get_fe_collection(),
      update_flags,
      crystals_data->get_n_slips()),
    gCP::AssemblyData::QuadraturePointHistory::Copy());
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::
update_local_quadrature_point_history(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  gCP::AssemblyData::QuadraturePointHistory::Scratch<dim>       &scratch,
  gCP::AssemblyData::QuadraturePointHistory::Copy               &)
{
  // Get the crystal identifier for the current cell
  const unsigned int crystal_id = cell->active_fe_index();

  //
  const std::vector<std::shared_ptr<QuadraturePointHistory<dim>>>
    local_quadrature_point_history =
      quadrature_point_history.get_data(cell);

  Assert(local_quadrature_point_history.size() == scratch.n_q_points,
         dealii::ExcInternalError());

  // Update the hp::FEValues instance to the values of the current cell
  scratch.hp_fe_values.reinit(cell);

  const dealii::FEValues<dim> &fe_values =
    scratch.hp_fe_values.get_present_fe_values();

  // Reset local data
  scratch.reset();

  // Compute the linear strain tensor at the quadrature points
  for (unsigned int slip_id = 0;
       slip_id < fe_field->get_n_slips();
       ++slip_id)
  {
    fe_values[fe_field->get_slip_extractor(crystal_id,
                                           slip_id)].get_function_values(
      fe_field->solution,
      scratch.slips_values[slip_id]);

    fe_values[fe_field->get_slip_extractor(crystal_id,
                                           slip_id)].get_function_values(
      fe_field->old_solution,
      scratch.old_slips_values[slip_id]);
  }

  // Loop over quadrature points
  for (const unsigned int q_point : fe_values.quadrature_point_indices())
  {
    local_quadrature_point_history[q_point]->update_values(
      q_point,
      scratch.slips_values,
      scratch.old_slips_values);
  } // Loop over quadrature points
}




} // namespace gCP



template void gCP::GradientCrystalPlasticitySolver<2>::assemble_jacobian();
template void gCP::GradientCrystalPlasticitySolver<3>::assemble_jacobian();

template void gCP::GradientCrystalPlasticitySolver<2>::assemble_local_jacobian(
  const typename dealii::DoFHandler<2>::active_cell_iterator  &,
  gCP::AssemblyData::Jacobian::Scratch<2>                     &,
  gCP::AssemblyData::Jacobian::Copy                           &);
template void gCP::GradientCrystalPlasticitySolver<3>::assemble_local_jacobian(
  const typename dealii::DoFHandler<3>::active_cell_iterator  &,
  gCP::AssemblyData::Jacobian::Scratch<3>                     &,
  gCP::AssemblyData::Jacobian::Copy                           &);

template void gCP::GradientCrystalPlasticitySolver<2>::copy_local_to_global_jacobian(
  const gCP::AssemblyData::Jacobian::Copy &);
template void gCP::GradientCrystalPlasticitySolver<3>::copy_local_to_global_jacobian(
  const gCP::AssemblyData::Jacobian::Copy &);

template void gCP::GradientCrystalPlasticitySolver<2>::assemble_residual();
template void gCP::GradientCrystalPlasticitySolver<3>::assemble_residual();

template void gCP::GradientCrystalPlasticitySolver<2>::assemble_local_residual(
  const typename dealii::DoFHandler<2>::active_cell_iterator  &,
  gCP::AssemblyData::Residual::Scratch<2>                     &,
  gCP::AssemblyData::Residual::Copy                           &);
template void gCP::GradientCrystalPlasticitySolver<3>::assemble_local_residual(
  const typename dealii::DoFHandler<3>::active_cell_iterator  &,
  gCP::AssemblyData::Residual::Scratch<3>                     &,
  gCP::AssemblyData::Residual::Copy                           &);

template void gCP::GradientCrystalPlasticitySolver<2>::copy_local_to_global_residual(
  const gCP::AssemblyData::Residual::Copy &);
template void gCP::GradientCrystalPlasticitySolver<3>::copy_local_to_global_residual(
  const gCP::AssemblyData::Residual::Copy &);

template void gCP::GradientCrystalPlasticitySolver<2>::update_quadrature_point_history();
template void gCP::GradientCrystalPlasticitySolver<3>::update_quadrature_point_history();

template void gCP::GradientCrystalPlasticitySolver<2>::update_local_quadrature_point_history(
  const typename dealii::DoFHandler<2>::active_cell_iterator  &,
  gCP::AssemblyData::QuadraturePointHistory::Scratch<2>       &,
  gCP::AssemblyData::QuadraturePointHistory::Copy             &);
template void gCP::GradientCrystalPlasticitySolver<3>::update_local_quadrature_point_history(
  const typename dealii::DoFHandler<3>::active_cell_iterator  &,
  gCP::AssemblyData::QuadraturePointHistory::Scratch<3>       &,
  gCP::AssemblyData::QuadraturePointHistory::Copy             &);