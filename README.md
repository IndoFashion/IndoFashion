## [IndoFashion: Apparel Classification for Indian Ethnic Clothes](https://indofashion.github.io/)

Indofashion dataset is the first large-scale ethnic dataset of over 106K images with 15 different categories for
fine-grained classification of Indian ethnic clothes. For more information, refer to the [paper](https://arxiv.org/abs/2104.02830). To access the dataset, please fill
out [this form](https://docs.google.com/forms/d/10Nke6m8MvCxP7hoJQ_k-mtiejbXtE0RliX9w_8pooLQ). We will provide you
script to download the dataset.

<p float="left">
  <img src="images/teaser.jpeg" width="100%" /> 
</p>


</br>
</br>

## Dataset Description

### Dataset Statistics

Our dataset consists of 106K images and 15 unique cloth
categories. For a fair evaluation, we ensure equal distribution of the classes in the validation and the test set consisting of 500 samples per class.
The dataset stats are listed below.

<p align='center'>Table 1: Class Categories.</p>
<table align='center'>

  <tr>
    <td><b>Gender</b></td>
    <td><b>Categories</b></td>
  </tr>
  <tr>
    <td align='center'>Women</td>
    <td align='center'> Saree, Women Kurta, Leggings & Salwar,
Palazzo, Lehenga, Dupatta, Blouse, Gown,
Dhoti Pants, Petticoats, Women Mojari</td>
  </tr>
  <tr>
    <td align='center'>Men</td>
    <td align='center'>Men Kurta, Nehru Jacket, Sherwani, Men
Mojari</td>

  </tr>
</table>

</br>
</br>

<p align='center'>Table 2: Dataset stats.</p>
<table align='center'>

  <tr>
    <td><b>Split</b></td>
    <td><b># Images</b></td>
  </tr>
  <tr>
    <td align='center'>Train</td>
    <td align='center'> 91,166</td>
  </tr>
  <tr>
    <td align='center'>Valid</td>
    <td align='center'>7,500</td>

  </tr>
  <tr>
    <td align='center'>Test </td>
    <td align='center'>7,500</td>
  </tr>
</table>

</br>

### Data Format

In the Indofashion dataset, training, validation and test sets are provided as JSON (JavaScript Object Notation) text files with the
following attributes for every data sample stored as a dictionary:

File Structure for train.json, val.json and test.json

```
{   "image_url": <image_url>,
    "image_path": <img_path>,
    "brand": <brand_name>,
    "product_title": <product_title>,
    "class_label": <class_label>
}
```

</br>
</br>

<p align='center'>Table 2: Attributes in Train/Validation,Test files.</p>

| Key   | Description                                                                      |
| -------- | ------------------------------------------------------------------------------|
| `image_url`    | URL used to download the image                     | 
| `image_path`          | Source path in dataset directory for the image                  |
| `brand`           | Brand name of the product                          |
| `product_title`       | Brief description of the product                           |
| `class_label`  | Label of the product                           |

</br>

*In cases where any of these attributes are not present, we substitute them with NA.


</br>

## Getting started

1. **Source Code:**   `$ git clone` this repo and install the Python dependencies from `requirements.txt`
2. **Dataset** Download the dataset by filling out the
   form [here](https://docs.google.com/forms/d/10Nke6m8MvCxP7hoJQ_k-mtiejbXtE0RliX9w_8pooLQ/edit).
3. **Train and Test** Coming Soon

</br>

## Citation

**If you find our dataset or paper useful for research , please include the following citation:**

```
@misc{rajput2021indofashion,
      title={IndoFashion : Apparel Classification for Indian Ethnic Clothes}, 
      author={Pranjal Singh Rajput and Shivangi Aneja},
      year={2021},
      eprint={2104.02830},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

</br>

## Contact Us

For any queries, please email us at indofashion.dataset@gmail.com. We will try to respond 
as soon as possible.
