#include <gCP/fe_field.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>


namespace gCP
{



template<int dim>
FEField<dim>::FEField(
  const dealii::Triangulation<dim>  &triangulation,
  const unsigned int                displacement_fe_degree,
  const unsigned int                slips_fe_degree)
:
displacement_fe_degree(displacement_fe_degree),
slips_fe_degree(slips_fe_degree),
dof_handler(triangulation),
flag_setup_extractors(true),
flag_setup_dofs(true),
flag_set_affine_constraints(true),
flag_setup_vectors(true)
{}



template<int dim>
void FEField<dim>::setup_extractors(const unsigned n_crystals, 
                                    const unsigned n_slips)
{
  this->n_crystals  = n_crystals;
  this->n_slips     = n_slips;

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
  for (dealii::types::material_id i = 0; i < n_crystals; ++i)
  {
    displacement_extractors.push_back(
      dealii::FEValuesExtractors::Vector(i*dim));

    std::vector<dealii::FEValuesExtractors::Scalar> 
      slip_extractors_per_crystal;

    for (unsigned int j = 0; j < n_slips; ++j)
      slip_extractors_per_crystal.push_back(
        dealii::FEValuesExtractors::Scalar(n_crystals * dim 
                                           + i * n_slips + j));

    slips_extractors.push_back(slip_extractors_per_crystal);
  }

  flag_setup_extractors = false;
}



template<int dim>
void FEField<dim>::setup_dofs()
{
  Assert(!flag_setup_extractors,
         dealii::ExcMessage("The method setup_extractors() has to be "
                            " called before setup_dofs()"))

  // The FESystem of the i-th crystal is divided into [ A | B ] with 
  // dimensions [ dim x n_crystals | n_slips x n_crystals ] where 
  //  A = FE_Nothing^dim     ... FE_Q^dim_i       ... FE_Nothing^dim 
  //  B = FE_Nothing^n_slips ... FE_Q^{n_slips}_i ... FE_Nothing^n_slips
  // If the displacment is continuous across crystalls then [ A | B ] has
  // the dimensiones [ dim | n_slips x n_crystals ] where A = FE_Q^dim
  for (dealii::types::material_id i = 0; i < n_crystals; ++i)
  {
    std::vector<const dealii::FiniteElement<dim>*>  finite_elements;

    // A
    if (true)
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
    for (auto finite_element: finite_elements)
      delete finite_element;
    finite_elements.clear();
  }

  // Distribute degrees of freedom based on the defined finite elements
  dof_handler.distribute_dofs(fe_collection);

  // Renumbering of the degrees of freedom
  dealii::DoFRenumbering::Cuthill_McKee(dof_handler);

  // Get the locally owned and relevant degrees of freedom of 
  // each processor
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  
  dealii::DoFTools::extract_locally_relevant_dofs(
    dof_handler,
    locally_relevant_dofs);

  // Initiate the hanging node constraints
  hanging_node_constraints.clear();
  {
    hanging_node_constraints.reinit(locally_relevant_dofs);
    
    dealii::DoFTools::make_hanging_node_constraints(
      dof_handler,
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
  {
    global_component_mapping.resize(n_crystals * (dim + n_slips));

    for (dealii::types::material_id i = 0; i < n_crystals; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
        global_component_mapping[i * dim + j] = j;
      
      for (unsigned int j = 0; j < n_slips; ++j)
        global_component_mapping[n_crystals * dim + i * n_slips + j] 
          = dim + j;
    }
  }

  // Modify flag because the dofs are setup
  flag_setup_dofs = false;
}



template<int dim>
void FEField<dim>::set_affine_constraints(
  const dealii::AffineConstraints<double> &affine_constraints)
{
    this->affine_constraints.merge(
      affine_constraints,
      dealii::AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
  
  flag_set_affine_constraints = false;
}



template<int dim>
void FEField<dim>::set_newton_method_constraints(
  const dealii::AffineConstraints<double> &newton_method_constraints)
{
    this->newton_method_constraints.merge(
      newton_method_constraints,
      dealii::AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
  
  flag_set_affine_constraints = false;
}



template<int dim>
void FEField<dim>::setup_vectors()
{
  solution.reinit(locally_relevant_dofs,
                  MPI_COMM_WORLD);

  distributed_vector.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            MPI_COMM_WORLD,
                            true);
  
  solution            = 0;
  distributed_vector  = 0;
}



} // gCP



template class gCP::FEField<2>;
template class gCP::FEField<3>;
