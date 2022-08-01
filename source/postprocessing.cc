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
Postprocessor<dim>::Postprocessor(
  std::shared_ptr<FEField<dim>>       &fe_field,
  std::shared_ptr<CrystalsData<dim>>  &crystals_data)
:
fe_field(fe_field),
crystals_data(crystals_data),
voigt_indices(6),
deviatoric_projector(
  dealii::identity_tensor<dim>() -
  1.0 / 3.0 *
  dealii::outer_product(
    dealii::unit_symmetric_tensor<dim>(),
    dealii::unit_symmetric_tensor<dim>())),
deviatoric_projector_3d(
  dealii::identity_tensor<3>() -
  1.0 / 3.0 *
  dealii::outer_product(
    dealii::unit_symmetric_tensor<3>(),
    dealii::unit_symmetric_tensor<3>()))
{
  voigt_indices[0] = std::make_pair<unsigned int, unsigned int>(0,0);
  voigt_indices[1] = std::make_pair<unsigned int, unsigned int>(1,1);
  voigt_indices[2] = std::make_pair<unsigned int, unsigned int>(2,2);
  voigt_indices[3] = std::make_pair<unsigned int, unsigned int>(1,2);
  voigt_indices[4] = std::make_pair<unsigned int, unsigned int>(0,2);
  voigt_indices[5] = std::make_pair<unsigned int, unsigned int>(0,1);
}



template <int dim>
std::vector<std::string>
Postprocessor<dim>::get_names() const
{
  std::vector<std::string> solution_names(dim, "Displacement");

  for (unsigned int slip_id = 0;
      slip_id < crystals_data->get_n_slips();
      ++slip_id)
    solution_names.emplace_back("Slip_" + std::to_string(slip_id));

  solution_names.emplace_back("EquivalentPlasticStrain");
  solution_names.emplace_back("EquivalentEdgeDislocationDensity");
  solution_names.emplace_back("EquivalentScrewDislocationDensity");
  solution_names.emplace_back("VonMisesStress");
  solution_names.emplace_back("VonMisesStrain");
  solution_names.emplace_back("Stress_11");
  solution_names.emplace_back("Stress_22");
  solution_names.emplace_back("Stress_33");
  solution_names.emplace_back("Stress_23");
  solution_names.emplace_back("Stress_13");
  solution_names.emplace_back("Stress_12");
  solution_names.emplace_back("Strain_11");
  solution_names.emplace_back("Strain_22");
  solution_names.emplace_back("Strain_33");
  solution_names.emplace_back("Strain_23*2");
  solution_names.emplace_back("Strain_13*2");
  solution_names.emplace_back("Strain_12*2");

  return solution_names;
}


template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
Postprocessor<dim>::get_data_component_interpretation()
  const
{
  std::vector<
    dealii::DataComponentInterpretation::DataComponentInterpretation>
      interpretation(
        dim,
        dealii::DataComponentInterpretation::component_is_part_of_vector);

  for (unsigned int slip_id = 0;
      slip_id < crystals_data->get_n_slips();
      ++slip_id)
    interpretation.push_back(
      dealii::DataComponentInterpretation::component_is_scalar);

  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);

  return interpretation;
}


template <int dim>
dealii::UpdateFlags
Postprocessor<dim>::get_needed_update_flags() const
{
  return dealii::update_values |
         dealii::update_gradients |
         dealii::update_quadrature_points;
}



template <int dim>
void Postprocessor<dim>::init(
  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law)
{
  this->hooke_law       = hooke_law;

  flag_init_was_called  = true;
}



template <int dim>
void Postprocessor<dim>::evaluate_vector_field(
  const dealii::DataPostprocessorInputs::Vector<dim>  &inputs,
  std::vector<dealii::Vector<double>>                 &computed_quantities) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The Postprocessor<dim> instance has"
                                 " not been initialized."));

  const typename dealii::DoFHandler<dim>::cell_iterator current_cell =
    inputs.template get_cell<dim>();

  const unsigned int material_id  = current_cell->material_id();

  const unsigned int n_q_points   = inputs.solution_values.size();

  const unsigned int n_components = fe_field->get_n_components();

  const unsigned int n_slips      = crystals_data->get_n_slips();

  const unsigned int n_crystals   = crystals_data->get_n_crystals();

  (void)n_components;

  Assert(inputs.solution_gradients.size() == n_q_points,
         dealii::ExcInternalError());

  Assert(computed_quantities.size() == n_q_points,
         dealii::ExcInternalError());

  Assert(inputs.solution_values[0].size() == n_components,
         dealii::ExcInternalError());

  // Reset
  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    for (unsigned int d = 0; d < computed_quantities[0].size(); ++d)
      computed_quantities[q_point](d) = 0.0;


  dealii::Tensor<2,dim>           displacement_gradient;
  dealii::SymmetricTensor<2,dim>  strain_tensor;
  dealii::SymmetricTensor<2,3>    strain_tensor_3d;
  dealii::SymmetricTensor<2,dim>  plastic_strain_tensor;
  dealii::SymmetricTensor<2,dim>  elastic_strain_tensor;
  dealii::SymmetricTensor<2,3>    stress_tensor;
  double                          equivalent_edge_dislocation_density;
  double                          equivalent_screw_dislocation_density;

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
  {
    // Reset
    displacement_gradient                 = 0.0;
    strain_tensor                         = 0.0;
    strain_tensor_3d                      = 0.0;
    plastic_strain_tensor                 = 0.0;
    elastic_strain_tensor                 = 0.0;
    stress_tensor                         = 0.0;
    equivalent_edge_dislocation_density   = 0.0;
    equivalent_screw_dislocation_density  = 0.0;

    if (fe_field->is_decohesion_allowed())
    {
      // Displacement
      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int crystal_id = 0;
              crystal_id < n_crystals; ++crystal_id)
      {
        computed_quantities[q_point](d) +=
            inputs.solution_values[q_point](d + dim * crystal_id);

        displacement_gradient[d] +=
          inputs.solution_gradients[q_point][d + dim * crystal_id];
      }

      strain_tensor     = dealii::symmetrize(displacement_gradient);

      strain_tensor_3d  = convert_2d_to_3d(strain_tensor);

      for (unsigned int slip_id = 0;
          slip_id < n_slips; ++slip_id)
        for (unsigned int crystal_id = 0;
            crystal_id < n_crystals; ++crystal_id)
        {
          // Slips
          computed_quantities[q_point](dim + slip_id) +=
            inputs.solution_values[q_point](
              dim * n_crystals + slip_id + n_slips * crystal_id);

          // Equivalent plastic strain
          computed_quantities[q_point](dim + n_slips) +=
            inputs.solution_values[q_point](
              dim * n_crystals + slip_id + n_slips * crystal_id);

          // Equivalent edge dislocation density
          equivalent_edge_dislocation_density +=
            std::pow(inputs.solution_gradients[q_point][
                dim * n_crystals + slip_id + n_slips * crystal_id] *
            crystals_data->get_slip_direction(crystal_id, slip_id), 2);

          // Equivalent screw dislocation density
          equivalent_screw_dislocation_density +=
            std::pow(inputs.solution_gradients[q_point][
                dim * n_crystals + slip_id + n_slips * crystal_id] *
            crystals_data->get_slip_orthogonal(crystal_id, slip_id), 2);

          // Plastic strain tensor
          plastic_strain_tensor +=
            inputs.solution_values[q_point](
                dim * n_crystals + slip_id + n_slips * crystal_id) *
            crystals_data->get_symmetrized_schmid_tensor(
              crystal_id, slip_id);
        }

      computed_quantities[q_point](dim + n_slips + 1) =
        std::sqrt(equivalent_edge_dislocation_density);

      computed_quantities[q_point](dim + n_slips + 2) =
        std::sqrt(equivalent_screw_dislocation_density);

      elastic_strain_tensor =  strain_tensor - plastic_strain_tensor;

      stress_tensor =
        hooke_law->get_stiffness_tetrad_3d(material_id) *
        convert_2d_to_3d(elastic_strain_tensor);

      // Von-Mises stress
      computed_quantities[q_point](dim + n_slips + 3) =
        get_von_mises_stress(stress_tensor);

      // Von-Mises plastic strain
      computed_quantities[q_point](dim + n_slips + 4) =
        get_von_mises_plastic_strain(plastic_strain_tensor);

      // Stress components
      for (unsigned int i = 0; i < voigt_indices.size(); ++i)
        computed_quantities[q_point](dim + n_slips + 5 + i) =
          stress_tensor[voigt_indices[i].first][voigt_indices[i].second];

      // Strain components
      for (unsigned int i = 0; i < voigt_indices.size(); ++i)
        computed_quantities[q_point](dim + n_slips + 11 + i) =
          (i < 3 ? 1.0 : 2.0) *
          strain_tensor_3d[voigt_indices[i].first][voigt_indices[i].second];
    }
    else
    {
      // Displacement
      for (unsigned int d = 0; d < dim; ++d)
      {
        computed_quantities[q_point](d) =
          inputs.solution_values[q_point](d);

        displacement_gradient[d] =
          inputs.solution_gradients[q_point][d];
      }

      strain_tensor     = dealii::symmetrize(displacement_gradient);

      strain_tensor_3d  = convert_2d_to_3d(strain_tensor);

      for (unsigned int slip_id = 0;
          slip_id < n_slips; ++slip_id)
        for (unsigned int crystal_id = 0;
            crystal_id < n_crystals; ++crystal_id)
        {
          // Slips
          computed_quantities[q_point](dim + slip_id) +=
              inputs.solution_values[q_point](
                dim + slip_id + n_slips * crystal_id);

          // Equivalent plastic strain
          computed_quantities[q_point](dim + n_slips) +=
              inputs.solution_values[q_point](
                dim + slip_id + n_slips * crystal_id);

          // Equivalent edge dislocation density
          equivalent_edge_dislocation_density +=
            std::pow(inputs.solution_gradients[q_point][
                dim + slip_id + n_slips * crystal_id] *
            crystals_data->get_slip_direction(crystal_id, slip_id), 2);

          // Equivalent screw dislocation density
          equivalent_screw_dislocation_density +=
            std::pow(inputs.solution_gradients[q_point][
                dim + slip_id + n_slips * crystal_id] *
            crystals_data->get_slip_orthogonal(crystal_id, slip_id), 2);

          // Plastic strain tensor
          plastic_strain_tensor +=
            inputs.solution_values[q_point](
                dim + slip_id + n_slips * crystal_id) *
            crystals_data->get_symmetrized_schmid_tensor(
              crystal_id, slip_id);
        }

      computed_quantities[q_point](dim + n_slips + 1) =
        std::sqrt(equivalent_edge_dislocation_density);

      computed_quantities[q_point](dim + n_slips + 2) =
        std::sqrt(equivalent_screw_dislocation_density);

      elastic_strain_tensor =  strain_tensor - plastic_strain_tensor;

      stress_tensor =
        hooke_law->get_stiffness_tetrad_3d(material_id) *
        convert_2d_to_3d(elastic_strain_tensor);

      // Von-Mises stress
      computed_quantities[q_point](dim + n_slips + 3) =
        get_von_mises_stress(stress_tensor);

      // Von-Mises plastic strain
      computed_quantities[q_point](dim + n_slips + 4) =
        get_von_mises_plastic_strain(plastic_strain_tensor);

      // Stress components
      for (unsigned int i = 0; i < voigt_indices.size(); ++i)
        computed_quantities[q_point](dim + n_slips + 5 + i) =
          stress_tensor[voigt_indices[i].first][voigt_indices[i].second];

      // Strain components
      for (unsigned int i = 0; i < voigt_indices.size(); ++i)
        computed_quantities[q_point](dim + n_slips + 11 + i) =
          (i < 3 ? 1.0 : 2.0) *
          strain_tensor_3d[voigt_indices[i].first][voigt_indices[i].second];
    }
  }
}



template <int dim>
dealii::SymmetricTensor<2,3>
Postprocessor<dim>::convert_2d_to_3d(
  dealii::SymmetricTensor<2,dim> symmetric_tensor) const
{
  dealii::SymmetricTensor<2,3> symmetric_tensor_in_3d;

  if constexpr(dim == 3)
    symmetric_tensor_in_3d = symmetric_tensor;
  else if constexpr(dim ==2)
  {
    symmetric_tensor_in_3d[0][0] = symmetric_tensor[0][0];
    symmetric_tensor_in_3d[1][1] = symmetric_tensor[1][1];
    symmetric_tensor_in_3d[0][1] = symmetric_tensor[0][1];
  }
  else
    Assert(false, dealii::ExcNotImplemented());

  return (symmetric_tensor_in_3d);
}



template <int dim>
double Postprocessor<dim>::get_von_mises_stress(
  dealii::SymmetricTensor<2,3> stress_tensor_in_3d) const
{
  const dealii::SymmetricTensor<2,3> deviatoric_stress_tensor =
    deviatoric_projector_3d * stress_tensor_in_3d;

  double von_mises_stress =
    deviatoric_stress_tensor * deviatoric_stress_tensor;

  return (std::sqrt(3.0 / 2.0 * von_mises_stress));
}



template <int dim>
double Postprocessor<dim>::get_von_mises_plastic_strain(
  dealii::SymmetricTensor<2,dim> strain_tensor) const
{
  const dealii::SymmetricTensor<2,dim> deviatoric_strain_tensor =
    deviatoric_projector * strain_tensor;

  double von_mises_strain =
    deviatoric_strain_tensor * deviatoric_strain_tensor;

  return (std::sqrt(2.0 / 3.0 * von_mises_strain));
}



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
shear_at_upper_boundary(shear_at_upper_boundary),
upper_boundary_id(upper_boundary_id),
width(width),
flag_init_was_called(false)
{
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

  flag_init_was_called = true;
}



template <int dim>
void SimpleShear<dim>::compute_data(const double time)
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The SimpleShear<dim>"
                                  " instance has not been"
                                  " initialized."));

  compute_stress_12_at_boundary();

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
void SimpleShear<dim>::output_data_to_file(
  std::ostream &file) const
{
  table_handler.write_text(
    file,
    dealii::TableHandler::TextOutputFormat::org_mode_table);
}



} // namespace Postprocessing



} // namespace gCP



template class gCP::Postprocessing::Postprocessor<2>;
template class gCP::Postprocessing::Postprocessor<3>;

template class gCP::Postprocessing::SimpleShear<2>;
template class gCP::Postprocessing::SimpleShear<3>;
