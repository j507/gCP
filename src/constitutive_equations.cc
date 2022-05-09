#include <gCP/constitutive_equations.h>

#include <deal.II/base/symmetric_tensor.h>

namespace gCP
{



namespace ConstitutiveEquations
{



template<int dim>
HookeLaw<dim>::HookeLaw(const double youngs_modulus,
                        const double poissons_ratio)
:
crystal_system(CrystalSystem::Isotropic),
C_1111(youngs_modulus * poissons_ratio /
       ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
       +
       youngs_modulus / (1.0 + poissons_ratio)),
C_1212(youngs_modulus / 2.0 / (1.0 + poissons_ratio)),
C_1122(youngs_modulus * poissons_ratio /
       ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))),
flag_init_was_called(false)
{}



template<int dim>
HookeLaw<dim>::HookeLaw(
  const std::shared_ptr<CrystalsData<dim>> &crystals_data,
  const double                             C_1111,
  const double                             C_1212,
  const double                             C_1122)
:
crystals_data(crystals_data),
crystal_system(CrystalSystem::Cubic),
C_1111(C_1111),
C_1212(C_1212),
C_1122(C_1122),
flag_init_was_called(false)
{}



template<int dim>
void HookeLaw<dim>::init()
{
  for (unsigned int i = 0; i < dim; i++)
    for (unsigned int j = 0; j < dim; j++)
      for (unsigned int k = 0; k < dim; k++)
        for (unsigned int l = 0; l < dim; l++)
          if (i == j && j == k && k == l)
            reference_stiffness_tetrad[i][j][k][l] = C_1111;
          else if (i == k && j == l)
            reference_stiffness_tetrad[i][j][k][l] = C_1212;
          else if (i == j && k == l)
            reference_stiffness_tetrad[i][j][k][l] = C_1122;

  switch (crystal_system)
  {
  case CrystalSystem::Isotropic:
    break;

  case CrystalSystem::Cubic:
    {
      AssertThrow(crystals_data->is_initialized(),
                  dealii::ExcMessage("The underlying CrystalsData<dim>"
                                     " instance has not been "
                                     " initialized."));

      for (unsigned int crystal_id = 0;
           crystal_id < crystals_data->get_n_slips();
           crystal_id++)
        {
          dealii::SymmetricTensor<4,dim> stiffness_tetrad;

          dealii::Tensor<2,dim> rotation_tensor =
            crystals_data->get_rotation_tensor(crystal_id);

          for (unsigned int i = 0; i < dim; i++)
            for (unsigned int j = 0; j < dim; j++)
              for (unsigned int k = 0; k < dim; k++)
                for (unsigned int l = 0; l < dim; l++)
                  for (unsigned int o = 0; o < dim; o++)
                    for (unsigned int p = 0; p < dim; p++)
                      for (unsigned int q = 0; q < dim; q++)
                        for (unsigned int r = 0; r < dim; r++)
                          stiffness_tetrad[i][j][k][l] =
                            rotation_tensor[i][o] *
                            rotation_tensor[j][p] *
                            rotation_tensor[k][q] *
                            rotation_tensor[l][r] *
                            reference_stiffness_tetrad[o][p][q][r];

          stiffness_tetrads.push_back(stiffness_tetrad);
        }
    }
    break;

  default:
    break;
  }

  flag_init_was_called = true;
}


template<int dim>
const dealii::SymmetricTensor<2,dim> HookeLaw<dim>::
compute_stress_tensor(
  const dealii::SymmetricTensor<2,dim> strain_tensor_values) const
{
  return reference_stiffness_tetrad * strain_tensor_values;
}



template<int dim>
const dealii::SymmetricTensor<2,dim> HookeLaw<dim>::
compute_stress_tensor(
  const unsigned int                    crystal_id,
  const dealii::SymmetricTensor<2,dim>  strain_tensor_values) const
{
  AssertThrow(crystals_data.get() != nullptr,
              dealii::ExcMessage("This overloaded method requires a "
                                 "constructor called where a "
                                 "CrystalsData<dim> instance is "
                                 "passed as a std::shared_ptr"))

  dealii::ExcIndexRangeType<int>(crystal_id,
                                 0,
                                 crystals_data->get_n_crystals());

  return stiffness_tetrads[crystal_id] * strain_tensor_values;
}



} // ConstitutiveEquations



} // gCP



template class gCP::ConstitutiveEquations::HookeLaw<2>;
template class gCP::ConstitutiveEquations::HookeLaw<3>;
