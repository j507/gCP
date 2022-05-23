#include <gCP/assembly_data.h>
#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace gCP
{



template <int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_jacobian()
{
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

  // Create instances of the quadrature collection classes
  dealii::hp::QCollection<dim>    quadrature_collection;

  dealii::hp::QCollection<dim-1>  face_quadrature_collection;

  // Initiate the quadrature formula for exact numerical integration
  const dealii::QGauss<dim>       quadrature_formula(3);

  const dealii::QGauss<dim-1>     face_quadrature_formula(3);

  // Add the initiated quadrature formulas to the collections
  quadrature_collection.push_back(quadrature_formula);

  face_quadrature_collection.push_back(face_quadrature_formula);

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
      update_flags),
    gCP::AssemblyData::Jacobian::Copy(
      fe_field->get_fe_collection().max_dofs_per_cell()));

  // Compress global data
  jacobian.compress(dealii::VectorOperation::add);
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_local_jacobian(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  gCP::AssemblyData::Jacobian::Scratch<dim>                     &scratch,
  gCP::AssemblyData::Jacobian::Copy                             &data)
{
  // Get the crystal identifier for the current cell
  const unsigned int crystal_id = cell->active_fe_index();

  // Reset local data
  data.local_matrix = 0.0;

  // Update the hp::FEValues instance to the values of the current cell
  scratch.hp_fe_values.reinit(cell);

  const dealii::FEValues<dim> &fe_values =
    scratch.hp_fe_values.get_present_fe_values();

  const dealii::SymmetricTensor<4,dim> stiffness_tetrad =
    hooke_law->get_stiffness_tetrad(crystal_id);

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.sym_grad_phi[i] =
        fe_values[fe_field->get_displacement_extractor(crystal_id)].symmetric_gradient(i,q);
    }

    // Mapped quadrature weight
    double dv = fe_values.JxW(q);

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
      {
        data.local_matrix(i,j) -=
          scratch.sym_grad_phi[i] * -1.0 *
          stiffness_tetrad *
          scratch.sym_grad_phi[j] *
          dv;
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

  // Create instances of the quadrature collection classes
  dealii::hp::QCollection<dim>    quadrature_collection;

  dealii::hp::QCollection<dim-1>  face_quadrature_collection;

  // Initiate the quadrature formula for exact numerical integration
  const dealii::QGauss<dim>       quadrature_formula(3);

  const dealii::QGauss<dim-1>     face_quadrature_formula(3);

  // Add the initiated quadrature formulas to the collections
  quadrature_collection.push_back(quadrature_formula);

  face_quadrature_collection.push_back(face_quadrature_formula);

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
      face_update_flags),
    gCP::AssemblyData::Residual::Copy(
      fe_field->get_fe_collection().max_dofs_per_cell()));

  // Compress global data
  residual.compress(dealii::VectorOperation::add);

  residual_norm = residual.l2_norm();
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_local_residual(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  gCP::AssemblyData::Residual::Scratch<dim>                     &scratch,
  gCP::AssemblyData::Residual::Copy                             &data)
{
  // Get the crystal identifier for the current cell
  const unsigned int crystal_id = cell->active_fe_index();

  // Reset local data
  data.local_rhs                          = 0.0;
  data.local_matrix_for_inhomogeneous_bcs = 0.0;

  // Local to global mapping of the indices of the degrees of freedom
  cell->get_dof_indices(data.local_dof_indices);

  // Update the hp::FEValues instance to the values of the current cell
  scratch.hp_fe_values.reinit(cell);

  const dealii::FEValues<dim> &fe_values =
    scratch.hp_fe_values.get_present_fe_values();

  // Compute the linear strain tensor at the quadrature points
  fe_values[fe_field->get_displacement_extractor(crystal_id)].get_function_symmetric_gradients(
    fe_field->solution,
    scratch.strain_tensor_values);

  // Compute the supply term at the quadrature points
  supply_term->value_list(
    fe_values.get_quadrature_points(),
    scratch.supply_term_values);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Compute the stress tensor at the quadrature points
    scratch.stress_tensor_values[q] =
      hooke_law->get_stress_tensor(crystal_id,
                                         scratch.strain_tensor_values[q]);

    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i] =
        fe_values[fe_field->get_displacement_extractor(crystal_id)].value(i,q);

      scratch.sym_grad_phi[i] =
        fe_values[fe_field->get_displacement_extractor(crystal_id)].symmetric_gradient(i,q);
    }

    // Mapped quadrature weight
    double dv = fe_values.JxW(q);

    // Loop over the degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs(i) +=
        (scratch.sym_grad_phi[i] *
         scratch.stress_tensor_values[q] * 0.0
         -
         scratch.phi[i] * -1.0 *
         scratch.supply_term_values[q]) *
         dv;
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
        for (unsigned int q = 0; q < scratch.n_face_q_points; ++q)
        {
          // Extract the test function's values at the face quadrature points
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            scratch.face_phi[i] =
              fe_face_values[fe_field->get_displacement_extractor(crystal_id)].value(i,q);

          // Mapped quadrature weight
          double da = fe_face_values.JxW(q);

          // Loop over degrees of freedom
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            data.local_rhs(i) -=
              scratch.face_phi[i] *
              scratch.neumann_boundary_values[q] *
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

  // Create instances of the quadrature collection classes
  dealii::hp::QCollection<dim>    quadrature_collection;

  // Initiate the quadrature formula for exact numerical integration
  const dealii::QGauss<dim>       quadrature_formula(3);

  // Add the initiated quadrature formulas to the collections
  quadrature_collection.push_back(quadrature_formula);

  // Set up the lambda function for the local assembly operation
  auto worker = [this](
    const CellIterator                                      &cell,
    gCP::AssemblyData::QuadraturePointHistory::Scratch<dim> &scratch,
    gCP::AssemblyData::QuadraturePointHistory::Copy         &data)
  {
    this->update_local_quadrature_point_history(cell, scratch, data);
  };

  // Set up the lambda function for the copy local to global operation
  auto copier = [this](const gCP::AssemblyData::Residual::Copy  &data)
  {
    this->copy_local_to_global_residual(data);
  };

  // Define the update flags for the FEValues instances
  const dealii::UpdateFlags update_flags  =
    dealii::update_default;

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
      update_flags),
    gCP::AssemblyData::QuadraturePointHistory::Copy());
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::
update_local_quadrature_point_history(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  gCP::AssemblyData::QuadraturePointHistory::Scratch<dim>       &scratch,
  gCP::AssemblyData::QuadraturePointHistory::Copy               &data)
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
       slip_id < fe_values->get_n_slips();
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
  const typename dealii::DoFHandler<2>::active_cell_iterator                       &,
  gCP::AssemblyData::Jacobian::Scratch<2>  &,
  gCP::AssemblyData::Jacobian::Copy        &);
template void gCP::GradientCrystalPlasticitySolver<3>::assemble_local_jacobian(
  const typename dealii::DoFHandler<3>::active_cell_iterator                       &,
  gCP::AssemblyData::Jacobian::Scratch<3>  &,
  gCP::AssemblyData::Jacobian::Copy        &);

template void gCP::GradientCrystalPlasticitySolver<2>::copy_local_to_global_jacobian(
  const gCP::AssemblyData::Jacobian::Copy &);
template void gCP::GradientCrystalPlasticitySolver<3>::copy_local_to_global_jacobian(
  const gCP::AssemblyData::Jacobian::Copy &);

template void gCP::GradientCrystalPlasticitySolver<2>::assemble_residual();
template void gCP::GradientCrystalPlasticitySolver<3>::assemble_residual();

template void gCP::GradientCrystalPlasticitySolver<2>::assemble_local_residual(
  const typename dealii::DoFHandler<2>::active_cell_iterator                       &,
  gCP::AssemblyData::Residual::Scratch<2>  &,
  gCP::AssemblyData::Residual::Copy        &);
template void gCP::GradientCrystalPlasticitySolver<3>::assemble_local_residual(
  const typename dealii::DoFHandler<3>::active_cell_iterator                       &,
  gCP::AssemblyData::Residual::Scratch<3>  &,
  gCP::AssemblyData::Residual::Copy        &);

template void gCP::GradientCrystalPlasticitySolver<2>::copy_local_to_global_residual(
  const gCP::AssemblyData::Residual::Copy &);
template void gCP::GradientCrystalPlasticitySolver<3>::copy_local_to_global_residual(
  const gCP::AssemblyData::Residual::Copy &);
