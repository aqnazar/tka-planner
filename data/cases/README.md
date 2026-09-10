# Cases

One folder per case, named with an anonymised case id and nothing else:

```
data/cases/
  P009/
    FD1Left.stl     the femur, as segmented
    TD1Left.stl     the tibia, as segmented
```

Use `FD1Right.stl` and `TD1Right.stl` for a right knee. The planner reads millimetres
in whatever frame the segmentation was exported in and builds its own anatomical frame
from the geometry, so the scanner's coordinates do not have to be corrected first.

The meshes themselves are ignored by git, and that is deliberate. A folder name is an
anonymised id; a scan is patient data. The map from case id to a real person stays on
the machine that made it and never enters this repository.
