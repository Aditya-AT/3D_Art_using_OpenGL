
# 3D Christmas Scene Renderer

## Overview

This project is a culmination of the learning objectives from the Computer Graphics course, focusing on building a complex 3D scene. The chosen theme for the final programming assignment is a festive Christmas scene featuring a detailed Christmas tree surrounded by gifts, utilizing various aspects of 3D rendering including transformations, camera manipulation, shading, and texture mapping.

## Features

- **3D Models**: Utilizes basic geometrical shapes like cubes (for gifts), cones, and cylinders (for the Christmas tree) to construct the scene.
- **Camera System**: Implements non-default camera positioning to capture the festive setup from the best angles, enhancing the visual appeal.
- **Lighting and Shading**: Incorporates at least one point light source to simulate a cozy, festive glow, with Phong shading to give depth and realism to the objects.
- **Texture Mapping**: Uses both image-based and procedural textures to add detail to the Christmas tree and the gifts, making them look as realistic as possible.

## Installation

Clone the repository to your local machine using:

```bash
git clone <repository-url>
```

## Usage

To view the Christmas scene, run the main rendering script after navigating to the project directory:

```bash
python main.py
```

## Technologies Used

- Python: For scripting the 3D scene setup and rendering logic.
- OpenGL: For handling the graphical rendering.
- GLSL: For shader programming to achieve desired material and lighting effects.

## Project Structure

- `shapes.py`: Contains definitions for basic geometric shapes used in the scene.
- `main.py`: Entry point for rendering the Christmas scene.
- `rit_window.py`: Handles the windowing and interaction aspects.

## Contributions

Feel free to fork this project and submit pull requests with improvements or new features. All contributions that enhance or expand the project are welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
