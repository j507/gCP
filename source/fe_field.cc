#include <gCP/fe_field.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>



namespace gCP
{



template <int dim>
FEField<dim>::FEField(
  const dealii::Triangulation<dim> &triangulation,
  const unsigned int displacement_fe_degree,
  const unsigned int slips_fe_degree,
  const bool flag_allow_decohesion,
  const bool flag_use_single_block)
:
displacement_fe_degree(displacement_fe_degree),
slips_fe_degree(slips_fe_degree),
dof_handler(std::make_shared<dealii::DoFHandler<dim>>(triangulation)),
flag_allow_decohesion(flag_allow_decohesion),
flag_use_single_block(flag_use_single_block),
flag_setup_extractors_was_called(false),
flag_setup_dofs_was_called(false),
flag_affine_constraints_were_set(false),
flag_newton_method_constraints_were_set(false),
flag_setup_vectors_was_called(false)
{}



template <int dim>
FEField<dim>::FEField(const FEField<dim> &fe_field)
:
displacement_fe_degree(fe_field.displacement_fe_degree),
slips_fe_degree(fe_field.slips_fe_degree),
dof_handler(fe_field.dof_handler),
flag_allow_decohesion(fe_field.flag_allow_decohesion),
flag_use_single_block(fe_field.flag_use_single_block),
flag_setup_extractors_was_called(fe_field.flag_setup_dofs_was_called),
flag_setup_dofs_was_called(fe_field.flag_setup_dofs_was_called),
flag_affine_constraints_were_set(false),
flag_newton_method_constraints_were_set(false),
flag_setup_vectors_was_called(false)
{
  if (flag_setup_dofs_was_called)
  {
    n_crystals = fe_field.n_crystals;

    n_slips = fe_field.n_slips;

    displacement_extractors = fe_field.displacement_extractors;

    slips_extractors = fe_field.slips_extractors;

    fe_collection =
      dealii::hp::FECollection<dim>(fe_field.fe_collection);

    n_displacement_dofs = fe_field.n_displacement_dofs;

    n_plastic_slip_dofs = fe_field.n_plastic_slip_dofs;

    locally_owned_dofs = fe_field.locally_owned_dofs;

    locally_owned_displacement_dofs =
      fe_field.locally_owned_displacement_dofs;

    locally_owned_plastic_slip_dofs =
      fe_field.locally_owned_plastic_slip_dofs;

    locally_relevant_dofs = fe_field.locally_relevant_dofs;

    locally_owned_dofs_per_block =
      fe_field.locally_owned_dofs_per_block;

    locally_relevant_dofs_per_block =
      fe_field.locally_relevant_dofs_per_block;

    hanging_node_constraints.clear();
    {
      hanging_node_constraints.reinit(locally_relevant_dofs);

      dealii::DoFTools::make_hanging_node_constraints(
          *dof_handler,
          hanging_node_constraints);
    }
    hanging_node_constraints.close();

    affine_constraints.clear();
    {
      affine_constraints.reinit(locally_relevant_dofs);
      affine_constraints.merge(hanging_node_constraints);
    }
    affine_constraints.close();

    newton_method_constraints.clear();
    {
      newton_method_constraints.reinit(locally_relevant_dofs);
      newton_method_constraints.merge(hanging_node_constraints);
    }
    newton_method_constraints.close();

    global_component_mapping = fe_field.global_component_mapping;

    vector_dof_indices = fe_field.vector_dof_indices;

    scalar_dof_indices = fe_field.scalar_dof_indices;
  }
}



template <int dim>
void FEField<dim>::setup_extractors(const unsigned n_crystals,
                                    const unsigned n_slips)
{
  this->n_crystals = n_crystals;
  this->n_slips = n_slips;

  // displacement_extractors contains Vector extractors which can be
  // thought as the ComponentMasks of size n_crystals x (n_slips + dim)
  // [true  x dim  false x (n_crystals - 1) false x n_crystals x n_slips]
  // [false x dim  true x dim               false x (n_crystals - 2) ...]
  // and so on
  // slip_extractors_per_crystal contains for the first crystal
  // [false x dim x n_crystals  true   false  false ...]
  // [false x dim x n_crystals  false  true   false ...]
  // [false x dim x n_crystals  false  false  true  ...]
  // for the second crystal
  // [false x dim x n_crystals  false x n_slips  true   false  false ...]
  // [false x dim x n_crystals  false x n_slips  false  true   false ...]
  // [false x dim x n_crystals  false x n_slips  false  false  true  ...]
  // and so on
  // This is just a visual aid. They are not a vector of booleans!
  if (flag_allow_decohesion)
    for (dealii::types::material_id i = 0; i < n_crystals; ++i)
    {
      displacement_extractors.push_back(
          dealii::FEValuesExtractors::Vector(i * dim));

      std::vector<dealii::FEValuesExtractors::Scalar>
          slip_extractors_per_crystal;

      for (unsigned int j = 0; j < n_slips; ++j)
        slip_extractors_per_crystal.push_back(
            dealii::FEValuesExtractors::Scalar(
                n_crystals * dim + i * n_slips + j));

      slips_extractors.push_back(slip_extractors_per_crystal);
    }
  else
    for (dealii::types::material_id i = 0; i < n_crystals; ++i)
    {
      displacement_extractors.push_back(
          dealii::FEValuesExtractors::Vector(0));
      std::vector<dealii::FEValuesExtractors::Scalar>
          slip_extractors_per_crystal;

      for (unsigned int j = 0; j < n_slips; ++j)
        slip_extractors_per_crystal.push_back(
            dealii::FEValuesExtractors::Scalar(dim + i * n_slips + j));

      slips_extractors.push_back(slip_extractors_per_crystal);
    }

  flag_setup_extractors_was_called = true;
}



template <int dim>
void FEField<dim>::update_ghost_material_ids()
{
  Utilities::update_ghost_material_ids(*dof_handler);
}



template <int dim>
void FEField<dim>::setup_dofs()
{
  AssertThrow(flag_setup_extractors_was_called,
              dealii::ExcMessage("The method setup_extractors() has to "
                                  "be called before setup_dofs()"));

  // The FESystem of the i-th crystal is divided into [ A | B ] with
  // dimensions [ dim x n_crystals | n_slips x n_crystals ] where
  //  A = FE_Nothing^dim     ... FE_Q^dim_i       ... FE_Nothing^dim
  //  B = FE_Nothing^n_slips ... FE_Q^{n_slips}_i ... FE_Nothing^n_slips
  // If the displacment is continuous across crystalls then [ A | B ] has
  // the dimensiones [ dim | n_slips x n_crystals ] where A = FE_Q^dim
  for (dealii::types::material_id i = 0; i < n_crystals; ++i)
  {
    std::vector<const dealii::FiniteElement<dim> *> finite_elements;

    // A
    if (flag_allow_decohesion)
      for (dealii::types::material_id j = 0; j < n_crystals; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          if (i == j)
            finite_elements.push_back(
                new dealii::FE_Q<dim>(displacement_fe_degree));
          else
            finite_elements.push_back(
                new dealii::FE_Nothing<dim>());
    else
      for (unsigned int j = 0; j < dim; ++j)
        finite_elements.push_back(
            new dealii::FE_Q<dim>(displacement_fe_degree));

    // B
    for (dealii::types::material_id j = 0; j < n_crystals; ++j)
      for (unsigned int k = 0; k < n_slips; ++k)
        if (i == j)
          finite_elements.push_back(
              new dealii::FE_Q<dim>(slips_fe_degree));
        else
          finite_elements.push_back(
              new dealii::FE_Nothing<dim>());

    // Add [ A | B ] of the i-th crystal to the FECollection
    fe_collection.push_back(
        dealii::FESystem<dim>(
            finite_elements,
            std::vector<unsigned int>(finite_elements.size(), 1)));

    // Delete in order to avoid memory leaks
    for (auto finite_element : finite_elements)
      delete finite_element;
    finite_elements.clear();
  }

  // Distribute degrees of freedom based on the defined finite elements
  dof_handler->distribute_dofs(fe_collection);

  // Renumbering of the degrees of freedom
  dealii::DoFRenumbering::Cuthill_McKee(*dof_handler);

  std::vector<dealii::types::global_dof_index> dofs_per_block;

  if (n_slips > 0)
  {
    const int n_displacement_components =
      dim * ((flag_allow_decohesion) ? n_crystals : 1);

    const int n_total_components =
      n_displacement_components + n_slips * n_crystals;

    std::vector<unsigned int> block_component(n_total_components, 0);

    for (unsigned int i = n_displacement_components;
          i < block_component.size(); ++i)
    {
      block_component[i] = 1;
    }

    dealii::DoFRenumbering::component_wise(
      *dof_handler, block_component);

    dofs_per_block =
      dealii::DoFTools::count_dofs_per_fe_block(
        *dof_handler, block_component);

    n_displacement_dofs = dofs_per_block[0];

    n_plastic_slip_dofs = dofs_per_block[1];

    if (flag_use_single_block)
    {
      dofs_per_block =
        std::vector<dealii::types::global_dof_index>(1, this->n_dofs());
    }
  }
  else
  {
    dofs_per_block =
      std::vector<dealii::types::global_dof_index>(1, this->n_dofs());
  }

  // Get the locally owned and relevant degrees of freedom of
  // each processor
  locally_owned_dofs = dof_handler->locally_owned_dofs();

  locally_owned_dofs_per_block.push_back(
    locally_owned_dofs.get_view(0, dofs_per_block[0]));

  if (!flag_use_single_block)
  {
    locally_owned_dofs_per_block.push_back(
    locally_owned_dofs.get_view(
      dofs_per_block[0], dof_handler->n_dofs()));
  }
  else
  {
    Assert(dofs_per_block[0] == this->n_dofs(),
      dealii::ExcMessage("The number of degrees of freedom do not "
       "match"));
  }

  dealii::DoFTools::extract_locally_relevant_dofs(
      *dof_handler,
      locally_relevant_dofs);

  locally_relevant_dofs_per_block.push_back(
    locally_relevant_dofs.get_view(0, dofs_per_block[0]));

  if (!flag_use_single_block)
  {
    locally_relevant_dofs_per_block.push_back(
      locally_relevant_dofs.get_view(
        dofs_per_block[0], dof_handler->n_dofs()));
  }
  else
  {
    Assert(dofs_per_block[0] == this->n_dofs(),
      dealii::ExcMessage("The number of degrees of freedom do not "
       "match"));
  }

  // Initiate the hanging node constraints
  hanging_node_constraints.clear();
  {
    hanging_node_constraints.reinit(locally_relevant_dofs);

    dealii::DoFTools::make_hanging_node_constraints(
        *dof_handler,
        hanging_node_constraints);
  }
  hanging_node_constraints.close();

  affine_constraints.clear();
  {
    affine_constraints.reinit(locally_relevant_dofs);
    affine_constraints.merge(hanging_node_constraints);
  }
  affine_constraints.close();

  newton_method_constraints.clear();
  {
    newton_method_constraints.reinit(locally_relevant_dofs);
    newton_method_constraints.merge(hanging_node_constraints);
  }
  newton_method_constraints.close();

  // Scope initiating the crystal-local to global component mapping
  if (flag_allow_decohesion)
  {
    global_component_mapping.resize(n_crystals * (dim + n_slips));

    for (dealii::types::material_id i = 0; i < n_crystals; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
        global_component_mapping[i * dim + j] = j;

      for (unsigned int j = 0; j < n_slips; ++j)
        global_component_mapping[n_crystals * dim + i * n_slips + j] =
            dim + j;
    }
  }
  else
  {
    global_component_mapping.resize(dim + n_crystals * n_slips);

    for (dealii::types::material_id i = 0; i < n_crystals; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
        global_component_mapping[j] = j;

      for (unsigned int j = 0; j < n_slips; ++j)
        global_component_mapping[dim + i * n_slips + j] = dim + j;
    }
  }

  // Store the local degrees of freedom indices related to the
  // displacement and the slips in two separate std::set
  {
    std::vector<dealii::types::global_dof_index> local_dof_indices(
        fe_collection.max_dofs_per_cell());

    for (const auto &active_cell : dof_handler->active_cell_iterators())
    {
      if (active_cell->is_locally_owned())
      {
        active_cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0;
              i < local_dof_indices.size();
              ++i)
        {
          if (get_global_component(active_cell->material_id(), i) < dim)
          {
            vector_dof_indices.insert(local_dof_indices[i]);
          }
          else
          {
            scalar_dof_indices.insert(local_dof_indices[i]);
          }
        }
      }
    }
  }

  vector_dof_indices = dealii::Utilities::MPI::compute_set_union(
      vector_dof_indices,
      MPI_COMM_WORLD);

  scalar_dof_indices = dealii::Utilities::MPI::compute_set_union(
      scalar_dof_indices,
      MPI_COMM_WORLD);

  Assert(this->n_dofs() ==
              (vector_dof_indices.size() + scalar_dof_indices.size()),
          dealii::ExcMessage(
              "Number of degrees of freedom do not match! " +
              std::to_string(this->n_dofs()) + "!=" +
              std::to_string(vector_dof_indices.size() +
                            scalar_dof_indices.size())))


  locally_owned_displacement_dofs =  locally_owned_dofs;

  locally_owned_plastic_slip_dofs =  locally_owned_dofs;

  for (unsigned int crystal_id = 0;
        crystal_id < this->get_n_crystals(); crystal_id++)
  {
    const dealii::IndexSet extracted_dofs =
      dealii::DoFTools::extract_dofs(
        this->get_dof_handler(),
        this->get_fe_collection().component_mask(
          this->get_displacement_extractor(crystal_id)));

    locally_owned_plastic_slip_dofs.subtract_set(extracted_dofs);
  }

  locally_owned_plastic_slip_dofs.compress();

  locally_owned_displacement_dofs.subtract_set(
    locally_owned_plastic_slip_dofs);

  locally_owned_displacement_dofs.compress();

  // Modify flag because the dofs are setup
  flag_setup_dofs_was_called = true;
}



template <int dim>
void FEField<dim>::set_affine_constraints(
    const dealii::AffineConstraints<double> &affine_constraints)
{
  AssertThrow(flag_setup_dofs_was_called,
              dealii::ExcMessage("The method setup_dofs() has to be "
                                  "called before setting the constriants"));

  this->affine_constraints.merge(
      affine_constraints,
      dealii::AffineConstraints<double>::MergeConflictBehavior::right_object_wins);

  flag_affine_constraints_were_set = true;
}



template <int dim>
void FEField<dim>::set_newton_method_constraints(
    const dealii::AffineConstraints<double> &newton_method_constraints)
{
  AssertThrow(flag_setup_dofs_was_called,
              dealii::ExcMessage("The method setup_dofs() has to be "
                                  "called before setting the constriants"));

  this->newton_method_constraints.merge(
      newton_method_constraints,
      dealii::AffineConstraints<double>::MergeConflictBehavior::right_object_wins);

  flag_newton_method_constraints_were_set = true;
}



template <int dim>
void FEField<dim>::setup_vectors()
{
  AssertThrow(flag_setup_dofs_was_called,
              dealii::ExcMessage("The setup_dofs() method has to be "
                                  "called before the setup_vectors() "
                                  " method."))

  // BlockVector instances
  solution.reinit(
    locally_relevant_dofs_per_block,
    MPI_COMM_WORLD);

  old_solution.reinit(solution);

  old_old_solution.reinit(solution);

  distributed_vector.reinit(
    locally_owned_dofs_per_block,
    locally_relevant_dofs_per_block,
    MPI_COMM_WORLD,
    true);

  solution = 0.;

  old_solution = 0.;

  old_old_solution = 0.;

  distributed_vector = 0.;

  flag_setup_vectors_was_called = true;
}



template <int dim>
dealii::LinearAlgebraTrilinos::MPI::BlockVector
FEField<dim>::get_distributed_vector_instance(
  const dealii::LinearAlgebraTrilinos::MPI::BlockVector &vector) const
{
  dealii::LinearAlgebraTrilinos::MPI::BlockVector tmp;

  tmp.reinit(distributed_vector);

  tmp = vector;

  return tmp;
}



template <int dim>
void FEField<dim>::prepare_for_serialization_of_active_fe_indices()
{
  AssertThrow(flag_setup_dofs_was_called,
              dealii::ExcMessage("The setup_dofs() method has to be "
                                  "called before the setup_vectors() "
                                  " method."))

      dof_handler->prepare_for_serialization_of_active_fe_indices();
}



template <int dim>
void FEField<dim>::update_solution_vectors()
{
  old_old_solution = old_solution;

  old_solution = solution;
}



template <int dim>
void FEField<dim>::reset_all_affine_constraints()
{
  affine_constraints.clear();
  {
    affine_constraints.reinit(locally_relevant_dofs);
    affine_constraints.merge(hanging_node_constraints);
  }
  affine_constraints.close();

  newton_method_constraints.clear();
  {
    newton_method_constraints.reinit(locally_relevant_dofs);
    newton_method_constraints.merge(hanging_node_constraints);
  }
  newton_method_constraints.close();
}



template <int dim>
double FEField<dim>::get_l2_norm(
  const dealii::LinearAlgebraTrilinos::MPI::BlockVector &vector) const
{
  if (vector.has_ghost_elements())
  {
    dealii::LinearAlgebraTrilinos::MPI::BlockVector tmp =
      get_distributed_vector_instance(vector);

    return (tmp.l2_norm());
  }
  else
  {
    return (vector.l2_norm());
  }
}



template <int dim>
dealii::Vector<double> FEField<dim>::get_sub_l2_norms(
  const dealii::LinearAlgebraTrilinos::MPI::BlockVector &vector,
  const double factor) const
{
  AssertIsFinite(factor);

  dealii::LinearAlgebraTrilinos::MPI::BlockVector distributed_vector_tmp;

  distributed_vector_tmp.reinit(distributed_vector);

  distributed_vector_tmp = vector;

  dealii::Vector<double> l2_norms(2);

  const double l2_norm = distributed_vector_tmp.l2_norm();

  if (flag_use_single_block)
  {


    /*Assert(distributed_vector.locally_owned_size() ==
            vector.locally_owned_size(),
           dealii::ExcMessage("The vectors are not of the same size"))
      */

    double vector_squared_entries = 0.0;

    double scalar_squared_entries = 0.0;

    for (const unsigned int locally_owned_displacement_dof :
          locally_owned_displacement_dofs)
    {
      vector_squared_entries +=
        distributed_vector_tmp[locally_owned_displacement_dof] *
        distributed_vector_tmp[locally_owned_displacement_dof];
    }

    for (const unsigned int locally_owned_plastic_slip_dof :
          locally_owned_plastic_slip_dofs)
    {
      scalar_squared_entries +=
        distributed_vector_tmp[locally_owned_plastic_slip_dof] *
        distributed_vector_tmp[locally_owned_plastic_slip_dof];
    }

    vector_squared_entries =
        dealii::Utilities::MPI::sum(
          vector_squared_entries,
          MPI_COMM_WORLD);

    scalar_squared_entries =
        dealii::Utilities::MPI::sum(
          scalar_squared_entries,
          MPI_COMM_WORLD);

    l2_norms[0] = std::sqrt(vector_squared_entries);

    l2_norms[1] = std::sqrt(scalar_squared_entries);
  }
  else
  {
    l2_norms[0] = distributed_vector_tmp.block(0).l2_norm();

    l2_norms[1] = distributed_vector_tmp.block(1).l2_norm();
  }

  l2_norms[0] /= factor;

  auto to_string =
      [](const double number)
  {
    std::ostringstream out;
    out.precision(10);
    out << std::scientific << number;
    return std::move(out).str();
  };

  {
    const double tolerance_relaxation_factor = 1e4;

    Assert(
      std::fabs(l2_norm - l2_norms.l2_norm()) <
        std::numeric_limits<double>::epsilon() *
          tolerance_relaxation_factor,
      dealii::ExcMessage("The norms do not match (" +
        to_string(l2_norm) + ", " + to_string(l2_norms.l2_norm()) +
          ")"));

    (void)l2_norm;
    (void)to_string;
    (void)tolerance_relaxation_factor;
  }

  return l2_norms;
}



} // gCP



template class gCP::FEField<2>;
template class gCP::FEField<3>;
