#ifndef INCLUDE_CRYSTAL_DATA_H_
#define INCLUDE_CRYSTAL_DATA_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>

#include <deal.II/distributed/tria.h>


namespace gCP
{



template<int dim>
class CrystalsData
{
public:

  CrystalsData();

  void init(const dealii::Triangulation<dim>  &triangulation,
            const std::string                 crystal_orientations_file_name,
            const std::string                 slip_directions_file_name,
            const std::string                 slip_normals_file_name);

  const unsigned int &get_n_crystals() const;

  const unsigned int &get_n_slips() const;

  const dealii::Tensor<2,dim> &get_rotation_tensor(
    const unsigned int crystal_id) const;

  const dealii::Tensor<2,3> &get_3d_rotation_tensor(
    const unsigned int crystal_id) const;

  const dealii::Tensor<1,dim> &get_slip_direction(
    const unsigned int crystal_id,
    const unsigned int slip_id) const;

  const std::vector<dealii::Tensor<1,dim>> &get_slip_directions(
    const unsigned int crystal_id) const;

  const dealii::Tensor<1,dim> &get_slip_normal(
    const unsigned int crystal_id,
    const unsigned int slip_id) const;

  const std::vector<dealii::Tensor<1,dim>> &get_slip_normals(
    const unsigned int crystal_id) const;

  const dealii::Tensor<1,dim> &get_slip_orthogonal(
    const unsigned int crystal_id,
    const unsigned int slip_id) const;

  const dealii::Tensor<2,dim> &get_schmid_tensor(
    const unsigned int crystal_id,
    const unsigned int slip_id) const;

  const dealii::SymmetricTensor<2,dim> &get_symmetrized_schmid_tensor(
    const unsigned int crystal_id,
    const unsigned int slip_id) const;

  const std::vector<dealii::SymmetricTensor<2,dim>>
    &get_symmetrized_schmid_tensors(
      const unsigned int crystal_id) const;

  const bool &is_initialized() const;

private:
  dealii::ConditionalOStream                      pcout;

  unsigned int                                    n_crystals;

  unsigned int                                    n_slips;

  std::vector<dealii::Tensor<1,dim>>              euler_angles;

  std::vector<dealii::Tensor<2,dim>>              rotation_tensors;

  std::vector<dealii::Tensor<2,3>>                rotation_tensors_3d;

  std::vector<dealii::Tensor<1,dim>>              reference_slip_directions;

  std::vector<dealii::Tensor<1,dim>>              reference_slip_normals;

  std::vector<dealii::Tensor<1,dim>>              reference_slip_orthogonals;

  std::vector<std::vector<dealii::Tensor<1,dim>>> slip_directions;

  std::vector<std::vector<dealii::Tensor<1,dim>>> slip_normals;

  std::vector<std::vector<dealii::Tensor<1,dim>>> slip_orthogonals;

  std::vector<std::vector<dealii::Tensor<2,dim>>> schmid_tensors;

  std::vector<std::vector<dealii::SymmetricTensor<2,dim>>>
                                                  symmetrized_schmid_tensors;

  bool                                            flag_init_was_called;

  void count_n_crystals(const dealii::Triangulation<dim>  &triangulation);

  void read_and_store_data(
    const std::string crystal_orientations_file_name,
    const std::string slip_directions_file_name,
    const std::string slip_normals_file_name);

  void compute_rotation_matrices();

  void compute_3d_rotation_matrices();

  void compute_slip_systems();

  bool orthogonality_check(const dealii::Tensor<1,dim> a,
                           const dealii::Tensor<1,dim> b);
};



template <int dim>
inline const unsigned int
&CrystalsData<dim>::get_n_crystals() const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  return (n_crystals);
}



template <int dim>
inline const unsigned int
&CrystalsData<dim>::get_n_slips() const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  return (n_slips);
}



template <int dim>
inline const dealii::Tensor<2,dim>
&CrystalsData<dim>::get_rotation_tensor(const unsigned int crystal_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  AssertIndexRange(crystal_id, n_crystals);
  return (rotation_tensors[crystal_id]);
}



template <int dim>
inline const dealii::Tensor<2,3>
&CrystalsData<dim>::get_3d_rotation_tensor(const unsigned int crystal_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  AssertIndexRange(crystal_id, n_crystals);
  return (rotation_tensors_3d[crystal_id]);
}



template <int dim>
inline const dealii::Tensor<1,dim>
&CrystalsData<dim>::get_slip_direction(const unsigned int crystal_id,
                                       const unsigned int slip_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  AssertIndexRange(crystal_id, n_crystals);
  AssertIndexRange(slip_id, n_slips);
  return (slip_directions[crystal_id][slip_id]);
}



template <int dim>
inline const std::vector<dealii::Tensor<1,dim>>
&CrystalsData<dim>::get_slip_directions(const unsigned int crystal_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  AssertIndexRange(crystal_id, n_crystals);
  return (slip_directions[crystal_id]);
}



template <int dim>
inline const dealii::Tensor<1,dim>
&CrystalsData<dim>::get_slip_normal(const unsigned int crystal_id,
                                    const unsigned int slip_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  AssertIndexRange(crystal_id, n_crystals);
  AssertIndexRange(slip_id, n_slips);
  return (slip_normals[crystal_id][slip_id]);
}



template <int dim>
inline const std::vector<dealii::Tensor<1,dim>>
&CrystalsData<dim>::get_slip_normals(const unsigned int crystal_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  AssertIndexRange(crystal_id, n_crystals);
  return (slip_normals[crystal_id]);
}



template <int dim>
inline const dealii::Tensor<1,dim>
&CrystalsData<dim>::get_slip_orthogonal(const unsigned int crystal_id,
                                        const unsigned int slip_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  AssertIndexRange(crystal_id, n_crystals);
  AssertIndexRange(slip_id, n_slips);
  return (slip_orthogonals[crystal_id][slip_id]);
}



template <int dim>
inline const dealii::Tensor<2,dim>
&CrystalsData<dim>::get_schmid_tensor(const unsigned int crystal_id,
                                      const unsigned int slip_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  AssertIndexRange(crystal_id, n_crystals);
  AssertIndexRange(slip_id, n_slips);
  return (schmid_tensors[crystal_id][slip_id]);
}



template <int dim>
inline const dealii::SymmetricTensor<2,dim>
&CrystalsData<dim>::get_symmetrized_schmid_tensor(
  const unsigned int crystal_id,
  const unsigned int slip_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  AssertIndexRange(crystal_id, n_crystals);
  AssertIndexRange(slip_id, n_slips);
  return (symmetrized_schmid_tensors[crystal_id][slip_id]);
}



template <int dim>
inline const std::vector<dealii::SymmetricTensor<2,dim>>
&CrystalsData<dim>::get_symmetrized_schmid_tensors(
  const unsigned int crystal_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The CrystalsData<dim>"
                                  " instance has not been"
                                  " initialized."));
  AssertIndexRange(crystal_id, n_crystals);
  return (symmetrized_schmid_tensors[crystal_id]);
}


template <int dim>
inline const bool
&CrystalsData<dim>::is_initialized() const
{
  return (flag_init_was_called);
}


}  // namespace gCP



#endif /* INCLUDE_CRYSTAL_DATA_H_ */
