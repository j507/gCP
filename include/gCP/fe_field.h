#ifndef INCLUDE_FE_FIELD_H_
#define INCLUDE_FE_FIELD_H_

#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <gCP/utilities.h>



namespace gCP
{



/**
 * @brief
 *
 * @tparam dim
 */
template <int dim>
class FEField
{
public:
/**
 * @brief Construct a new FEField object
 *
 * @param triangulation
 * @param displacement_fe_degree
 * @param slips_fe_degree
 */
FEField(const dealii::Triangulation<dim>  &triangulation,
        const unsigned int                displacement_fe_degree,
        const unsigned int                slips_fe_degree,
        const bool                        flag_allow_decohesion = false,
        const bool                        flag_use_single_block = false);

/*!
 * @brief Construct a new FEField object
 *
 * @param fe_field
 * @todo Docu
 */
FEField(const FEField<dim> &fe_field);

/*!
  * @brief The solution vector
  *
  * @details It contains the nodal values of the displacements and slips
  * fields.
  */
dealii::LinearAlgebraTrilinos::MPI::BlockVector solution;

/*!
  * @brief The solution vector of the previous time-step
  *
  * @details It contains the nodal values of the displacements and slips
  * fields.
  */
dealii::LinearAlgebraTrilinos::MPI::BlockVector old_solution;

/*!
  * @brief The solution vector of two load steps ago
  *
  * @details It contains the nodal values of the displacements and slips
  * fields.
  */
dealii::LinearAlgebraTrilinos::MPI::BlockVector old_old_solution;

/*!
  * @brief The vector-valued finite element field's corresponding
  * distributed vector.
  *
  * @details It is used to initiate distributed vectors using the copy
  * reinit() method.
  */
dealii::LinearAlgebraTrilinos::MPI::BlockVector
                                            distributed_vector;


/**
 * @brief Set the up extractors object
 *
 * @param n_crystals
 * @param n_slips
 * @todo Docu
 */
void setup_extractors(const unsigned n_crystals,
                      const unsigned n_slips);

/*!
  * @brief
  *
  * @todo Docu
  */
void update_ghost_material_ids();

/*!
  * @brief Set ups the degress of freedom of the vector-valued
  * finite element field
  */
void setup_dofs();

/**
 * @brief Set the affine constraints object
 *
 * @param affine_constraints
 * @todo Docu
 * @attention This method is only a temporary solution
 */
void set_affine_constraints(
  const dealii::AffineConstraints<double> &affine_constraints);

/**
 * @brief Set the newton constraints object
 *
 * @param newton_method_constraints
 * @todo Docu
 * @attention This method is only a temporary solution
 */
void set_newton_method_constraints(
  const dealii::AffineConstraints<double> &newton_method_constraints);

/*!
  * @brief Initializes the vectors by calling their respective
  * reinit method.
  */
void setup_vectors();

/*!
  * @brief
  *
  * @todo Docu
  */
void update_solution_vectors();

/*!
  * @brief
  *
  * @todo Docu
  */
void reset_all_affine_constraints();

/**
 * @brief Get the displacement fe degree object
 *
 * @return const unsigned int
 * @todo Docu
 */
unsigned int get_displacement_fe_degree() const;

/**
 * @brief Get the slips fe degree object
 *
 * @return const unsigned int
 * @todo Docu
 */
unsigned int get_slips_fe_degree() const;

/**
 * @brief Get the n crystals object
 *
 * @return unsigned int
 * @todo Docu
 */
unsigned int get_n_crystals() const;

/**
 * @brief Get the n slips object
 *
 * @return unsinged
 * @todo Docu
 */
unsigned int get_n_slips() const;

/*!
  * @brief Returns a const reference to the @ref triangulation.
  */
const dealii::Triangulation<dim> &get_triangulation() const;

/*!
  * @brief Returns a const reference to the @ref dof_handler
  */
const dealii::DoFHandler<dim>& get_dof_handler() const;

/*!
  * @brief Returns a const reference to the @ref finite_element
  */
const dealii::hp::FECollection<dim>& get_fe_collection() const;

/*!
  * @brief Returns a const reference to the
  * @ref hanging_node_constraints
  */
const dealii::AffineConstraints<double>&
  get_hanging_node_constraints() const;

/*!
  * @brief Returns a const reference to the @ref affine_constraints
  */
const dealii::AffineConstraints<double>&
  get_affine_constraints() const;

/*!
  * @brief Returns a const reference to the @ref newton_constraints
  */
const dealii::AffineConstraints<double>&
  get_newton_method_constraints() const;

/*!
  * @brief Returns a const reference to the @ref locally_owned_dofs
  */
const dealii::IndexSet& get_locally_owned_dofs() const;

const dealii::IndexSet& get_locally_owned_displacement_dofs() const;

const dealii::IndexSet& get_locally_owned_plastic_slip_dofs() const;


const std::vector<dealii::IndexSet>&
  get_locally_owned_dofs_per_block() const;

/*!
  * @brief Returns a const reference to the
  * @ref locally_relevant_owned_dofs
  */
const dealii::IndexSet& get_locally_relevant_dofs() const;

const std::vector<dealii::IndexSet>&
  get_locally_relevant_dofs_per_block() const;

/**
 * @brief Returns the global component
 *
 * @param local_component The component index
 * @todo Docu
 */
unsigned int get_global_component(
  const unsigned int crystal_id,
  const unsigned int local_component) const;

/**
 * @brief Get the displacement extractor object
 *
 * @param crystal_id
 * @return const dealii::FEValuesExtractors::Vector&
 * @todo Docu
 */
const dealii::FEValuesExtractors::Vector&
  get_displacement_extractor(const unsigned int crystal_id) const;

/**
 * @brief Get the slip extractor object
 *
 * @param crystal_id
 * @param slip_id
 * @return const dealii::FEValuesExtractors::Scalar&
 * @todo Docu
 */
const dealii::FEValuesExtractors::Scalar&
  get_slip_extractor(const unsigned int crystal_id,
                      const unsigned int slip_id) const;

/*!
  * @brief Get the extractors object
  *
  * @return const std::pair<
  * const std::vector<dealii::FEValuesExtractors::Vector>,
  * const std::vector<std::vector<dealii::FEValuesExtractors::Scalar>>>
  * @todo Docu
  */
const std::pair<
  std::vector<dealii::FEValuesExtractors::Vector>,
  std::vector<std::vector<dealii::FEValuesExtractors::Scalar>>>
    get_extractors() const;

dealii::LinearAlgebraTrilinos::MPI::BlockVector
  get_distributed_vector_instance(
    const dealii::LinearAlgebraTrilinos::MPI::BlockVector
      &vector) const;

/*!
  * @brief
  *
  * @todo Docu
  */
void prepare_for_serialization_of_active_fe_indices();

/*!
  * @brief Returns the number of degrees of freedom.
  */
dealii::types::global_dof_index n_dofs() const;

dealii::types::global_dof_index get_n_displacement_dofs() const;

dealii::types::global_dof_index get_n_plastic_slip_dofs() const;

/*!
  * @brief Returns the number of components of the vector-valued
  * finite element field.
  */
unsigned int get_n_components() const;

double get_l2_norm(
  const dealii::LinearAlgebraTrilinos::MPI::BlockVector &vector) const;

dealii::Vector<double> get_sub_l2_norms(
  const dealii::LinearAlgebraTrilinos::MPI::BlockVector &vector,
  const double factor = 1.0) const;

/*!
  * @brief
  *
  * @return true
  * @return false
  * @todo Docu
  */
bool is_decohesion_allowed() const;

/*!
  * @brief
  *
  * @return true
  * @return false
  * @todo Docu
  */
bool is_initialized() const;

private:

/**
 * @brief
 *
 * @todo Docu
 */
const unsigned int                displacement_fe_degree;

/**
 * @brief
 *
 * @todo Docu
 */
const unsigned int                slips_fe_degree;

/**
 * @brief
 *
 * @todo Docu
 */
unsigned int                      n_crystals;

/**
 * @brief
 *
 * @todo Docu
 */
unsigned int                      n_slips;

/*!
  * @brief The DoFHandler<dim> instance of the  vector-valued
  * finite element field.
  */
std::shared_ptr<dealii::DoFHandler<dim>> dof_handler;

/*!
  * @brief The FECollection<dim> instance of the vector-valued
  * finite element field.
  *
  * @todo Further explain
  */
dealii::hp::FECollection<dim>     fe_collection;

dealii::types::global_dof_index n_displacement_dofs;

dealii::types::global_dof_index n_plastic_slip_dofs;

/*!
  * @brief The AffineConstraints<double> instance handling the
  * hanging nodes.
  */
dealii::AffineConstraints<double> hanging_node_constraints;

/*!
  * @brief The AffineConstraints<double> instance handling the
  * hanging nodes and the boundary conditions.
  */
dealii::AffineConstraints<double> affine_constraints;

/*!
  * @brief The AffineConstraints<double> mirrors @ref affine_constraints
  * with the exception that all inhomogeneous constriants due to
  * Dirichlet boundary conditions are zero'ed out.
  */
dealii::AffineConstraints<double> newton_method_constraints;

/*!
  * @brief The set of the degrees of freedom owned by the processor.
  */
dealii::IndexSet                  locally_owned_dofs;

dealii::IndexSet                  locally_owned_displacement_dofs;

dealii::IndexSet                  locally_owned_plastic_slip_dofs;

/*!
  * @brief The set of the degrees of freedom that are relevant for
  * the processor.
  */
dealii::IndexSet                  locally_relevant_dofs;

std::vector<dealii::IndexSet>     locally_owned_dofs_per_block;

std::vector<dealii::IndexSet>     locally_relevant_dofs_per_block;

/**
 * @brief
 *
 * @todo Docu
 */
std::vector<unsigned int>         global_component_mapping;

/*!
  * @brief
  *
  * @todo Docu
  */
std::set<unsigned int>            vector_dof_indices;

/*!
  * @brief
  *
  * @todo Docu
  */
std::set<unsigned int>            scalar_dof_indices;

/**
 * @brief
 *
 * @todo Docu
 */
std::vector<dealii::FEValuesExtractors::Vector>
                                  displacement_extractors;

/**
 * @brief
 *
 * @todo Docu
 */
std::vector<std::vector<dealii::FEValuesExtractors::Scalar>>
                                  slips_extractors;

/**
 * @brief
 *
 * @todo Docu
 */
bool                              flag_allow_decohesion;

/**
 * @brief
 *
 * @todo Docu
 */
bool                              flag_use_single_block;

/**
 * @brief
 *
 * @todo Docu
 */
bool                              flag_setup_extractors_was_called;

/**
 * @brief
 *
 * @todo Docu
 */
bool                              flag_setup_dofs_was_called;

/**
 * @brief
 *
 * @todo Docu
 */
bool                              flag_affine_constraints_were_set;

/**
 * @brief
 *
 * @todo Docu
 */
bool                              flag_newton_method_constraints_were_set;

/**
 * @brief
 *
 * @todo Docu
 */
bool                              flag_setup_vectors_was_called;
};



template <int dim>
inline unsigned int
FEField<dim>::get_displacement_fe_degree() const
{
  return (displacement_fe_degree);
}



template <int dim>
inline unsigned int
FEField<dim>::get_slips_fe_degree() const
{
  return (slips_fe_degree);
}



template <int dim>
inline unsigned int
FEField<dim>::get_n_crystals() const
{
  return (n_crystals);
}



template <int dim>
inline unsigned int
FEField<dim>::get_n_slips() const
{
  return (n_slips);
}



template <int dim>
inline const dealii::Triangulation<dim> &
FEField<dim>::get_triangulation() const
{
  return (dof_handler->get_triangulation());
}



template <int dim>
inline const dealii::DoFHandler<dim> &
FEField<dim>::get_dof_handler() const
{
  return (*dof_handler);
}



template <int dim>
inline const dealii::hp::FECollection<dim> &
FEField<dim>::get_fe_collection() const
{
  return (fe_collection);
}



template <int dim>
inline const dealii::AffineConstraints<double> &
FEField<dim>::get_hanging_node_constraints() const
{
  return (hanging_node_constraints);
}



template <int dim>
inline const dealii::AffineConstraints<double> &
FEField<dim>::get_affine_constraints() const
{
  return (affine_constraints);
}



template <int dim>
inline const dealii::AffineConstraints<double> &
FEField<dim>::get_newton_method_constraints() const
{
  return (newton_method_constraints);
}



template <int dim>
inline const dealii::IndexSet &
FEField<dim>::get_locally_owned_dofs() const
{
  return (locally_owned_dofs);
}



template <int dim>
inline const dealii::IndexSet &
FEField<dim>::get_locally_owned_displacement_dofs() const
{
  return (locally_owned_displacement_dofs);
}



template <int dim>
inline const dealii::IndexSet &
FEField<dim>::get_locally_owned_plastic_slip_dofs() const
{
  return (locally_owned_plastic_slip_dofs);
}



template <int dim>
inline const std::vector<dealii::IndexSet> &
FEField<dim>::get_locally_owned_dofs_per_block() const
{
  return (locally_owned_dofs_per_block);
}



template <int dim>
inline const dealii::IndexSet &
FEField<dim>::get_locally_relevant_dofs() const
{
  return (locally_relevant_dofs);
}



template <int dim>
inline const std::vector<dealii::IndexSet>&
FEField<dim>::get_locally_relevant_dofs_per_block() const
{
  return (locally_relevant_dofs_per_block);
}



template <int dim>
inline unsigned int
FEField<dim>::get_global_component(
  const unsigned int crystal_id,
  const unsigned int local_component) const
{
  return (global_component_mapping[fe_collection[crystal_id].system_to_component_index(local_component).first]);
}



template <int dim>
inline const dealii::FEValuesExtractors::Vector&
FEField<dim>::get_displacement_extractor(const unsigned int crystal_id) const
{
  return (displacement_extractors[crystal_id]);
}



template <int dim>
inline const dealii::FEValuesExtractors::Scalar&
FEField<dim>::get_slip_extractor(const unsigned int crystal_id,
                                 const unsigned int slip_id) const
{
  return (slips_extractors[crystal_id][slip_id]);
}



template <int dim>
inline const std::pair<
  std::vector<dealii::FEValuesExtractors::Vector>,
  std::vector<std::vector<dealii::FEValuesExtractors::Scalar>>>
    FEField<dim>::get_extractors() const
{
  return std::make_pair(displacement_extractors, slips_extractors);
}



template <int dim>
inline dealii::types::global_dof_index
FEField<dim>::n_dofs() const
{
  return (dof_handler->n_dofs());
}



template <int dim>
inline dealii::types::global_dof_index
FEField<dim>::get_n_displacement_dofs() const
{
  return (n_displacement_dofs);
}



template <int dim>
inline dealii::types::global_dof_index
FEField<dim>::get_n_plastic_slip_dofs() const
{
  return (n_plastic_slip_dofs);
}



template <int dim>
inline unsigned int
FEField<dim>::get_n_components() const
{
  return (fe_collection.n_components());
}



template <int dim>
inline bool
FEField<dim>::is_decohesion_allowed() const
{
  return (flag_allow_decohesion);
}



template <int dim>
inline bool
FEField<dim>::is_initialized() const
{
  return (flag_setup_vectors_was_called &&
          flag_affine_constraints_were_set &&
          flag_newton_method_constraints_were_set);
}



} // gCP



#endif /* INCLUDE_FE_FIELD_H_ */