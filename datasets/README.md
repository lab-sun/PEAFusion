### Dataset
Download the datasets and place them in the 'datasets' folder with the following structure:
- [PST900 dataset](https://github.com/ShreyasSkandanS/pst900_thermal_rgb)
- [FMB dataset](https://github.com/JinyuanLiu-CV/SegMiF)
- [MF dataset](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)

```shell
<datasets>
|-- <PSTdataset>
    |-- <train>
        |-- rgb
        |-- thermal
        |-- labels
        ...
    |-- <test>
        |-- rgb
        |-- thermal
        |-- labels
        ...
|-- <FMB_dataset>
    |-- <train>
        |-- color
        |-- Infrared
        |-- Label
        |-- Visible
        ...
    |-- <test>
        |-- color
        |-- Infrared
        |-- Label
        |-- Visible
        ...
|-- <MFdataset>
    |-- <images>
    |-- <labels>
    |-- train.txt
    |-- val.txt
    |-- test.txt
    ...
```
