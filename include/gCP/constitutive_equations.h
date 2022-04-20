#ifndef INCLUDE_CONSTITUTIVE_EQUATIONS_H_
#define INCLUDE_CONSTITUTIVE_EQUATIONS_H_



#include <deal.II/base/symmetric_tensor.h>



namespace gCP
{



namespace ConstitutiveEquations
{



template<int dim>
class HookeLaw
{
public:
  HookeLaw(const double youngs_modulus,
           const double poissons_ratio);

  void compute_stiffness_tetrad();

  const dealii::SymmetricTensor<4,dim> &get_stiffness_tetrad() const;

  const dealii::SymmetricTensor<2,dim> compute_stress_tensor(
    const dealii::SymmetricTensor<2,dim> strain_tensor_values) const;

private:
  const double                          bulk_modulus;

  const double                          shear_modulus;

  const dealii::SymmetricTensor<4,dim>  spherical_projector;

  const dealii::SymmetricTensor<4,dim>  deviatoric_projector;

  dealii::SymmetricTensor<4,dim>        stiffness_tetrad;
};



template <int dim>
inline const dealii::SymmetricTensor<4,dim>
&HookeLaw<dim>::get_stiffness_tetrad() const
{
  return (stiffness_tetrad);
}



template<int dim, int n_slips>
class ResolvedShearStressLaw
{

};




template<int dim, int n_slips>
class ScalarMicroscopicStressLaw
{

};



template<int dim, int n_slips>
class VectorMicroscopicStressLaw
{

};



template<int dim, int n_slips>
class MicroscopicTractionLaw
{

};



template<int dim, int n_slips>
class MicroscopicInterfaceTractionLaw
{

};



} // ConstitutiveEquations



} // gCP



#endif /* INCLUDE_CONSTITUTIVE_EQUATIONS_H_ */