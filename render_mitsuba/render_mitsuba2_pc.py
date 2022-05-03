import numpy as np
import sys, os, subprocess
import OpenEXR
import Imath
from PIL import Image, ImageChops
from plyfile import PlyData, PlyElement

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

# PATH_TO_MITSUBA2 = "/home/tolga/Codes/mitsuba2/build/dist/mitsuba"  # mitsuba exectuable
PATH_TO_MITSUBA2 = "/home/wangyida/Documents/gitfarm/mitsuba2/build/dist/mitsuba"  # mitsuba exectuable

# replaced by command line arguments
# PATH_TO_NPY = 'pcl_ex.npy' # the tensor to load

# note that sampler is changed to 'independent' and the ldrfilm is changed to hdrfilm
# NOTE satisfying resolution            <integer name="width" value="720"/>
xml_head = \
    """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
        <boolean name="hide_emitters" value="true"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="{},{},{}" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="360"/>
            <integer name="height" value="360"/>
            <string name="pixel_format" value="rgba"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

# I also use a smaller point size 0.013 for ShapeNet, 0.009 for SUNCG
xml_ball_segment = \
    """
    <shape type="sphere">
        <float name="radius" value="0.013"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
    """

# material for plastic
"""
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
"""

# material for dielectric 
"""
	<bsdf type="roughdielectric">
	    <string name="distribution" value="beckmann"/>
	    <float name="alpha" value="0.1"/>
	    <string name="int_ior" value="bk7"/>
	    <string name="ext_ior" value="air"/>
            <rgb name="specular_reflectance" value="{},{},{}"/>
	</bsdf>
"""

# material for metal
"""
	<bsdf type="roughconductor">
	    <string name="material" value="Ag"/>
	    <string name="distribution" value="beckmann"/>
	    <float name="alpha" value="0.4"/>
            <rgb name="specular_reflectance" value="{},{},{}"/>
	</bsdf>
"""
obj_mesh = \
    """
    <shape type="obj">
	<string name="filename" value="{}"/>
        <bsdf type="diffuse">
        </bsdf>
    </shape>
    """
    
    # A rectangular bottom plane
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="{}"/>
        </transform>
    </shape>
"""

xml_tail = \
    """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="{}"/>
        </transform>
    </shape>
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="7,7,7"/>
        </emitter>
    </shape>
</scene>
"""


# setting for Eye objects
"""
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="0.1,0.1,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="7,7,7"/>
        </emitter>
    </shape>
"""


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


# only for debugging reasons
def writeply(vertices, ply_file):
    sv = np.shape(vertices)
    points = []
    for v in range(sv[0]):
        vertex = vertices[v]
        points.append("%f %f %f\n" % (vertex[0], vertex[1], vertex[2]))
    print(np.shape(points))
    file = open(ply_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    end_header
    %s
    ''' % (len(vertices), "".join(points)))
    file.close()


# as done in https://gist.github.com/drakeguan/6303065
def ConvertEXRToJPG(exrfile, jpgfile):
    File = OpenEXR.InputFile(exrfile)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)

    rgb = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    rgb_sums = rgb[0] + rgb[1] + rgb[2]
    for i in range(3):
        rgb[i] = np.where(rgb[i] <= 0.0031308,
                          (rgb[i] * 12.92) * 255.0,
                          (1.055 * (rgb[i] ** (1.0 / 2.4)) - 0.055) * 255.0)
        """
        rgb[i] = np.where(rgb_sums <= 0.0031308,
                          255.0,
                          (1.055 * (rgb[i] ** (1.0 / 2.4)) - 0.055) * 255.0)
        """

    rgb8 = [Image.frombytes("F", Size, c.tostring()).convert("L") for c in rgb]
    rgb8 = Image.merge("RGB", rgb8)
    rgb8 = trim(rgb8)
    """
    It's for rendering with identical color
    """
    # rgb8 = [Image.fromarray(c.astype(int)) for c in rgb]
    rgb8.save(jpgfile, "JPEG", quality=95)


def main(argv):
    if (len(argv) < 2):
        print('filename to npy/ply is not passed as argument. terminated.')
        return

    pathToFile = argv[1]

    filename, file_extension = os.path.splitext(pathToFile)
    folder = os.path.dirname(pathToFile)
    filename = os.path.basename(pathToFile)

    # for the moment supports npy and ply
    if (file_extension == '.npy'):
        pclTime = np.load(pathToFile)
        pclTimeSize = np.shape(pclTime)
    elif (file_extension == '.npz'):
        pclTime = np.load(pathToFile)
        pclTime = pclTime['pred']
        pclTimeSize = np.shape(pclTime)
    elif (file_extension == '.ply'):
        ply = PlyData.read(pathToFile)
        vertex = ply['vertex']
        (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        pclTime = np.column_stack((x, y, z))
    elif (file_extension == '.list'):
        sys.path.append("../")
        from dataset import read_points
        with open(pathToFile) as file:
            tempname = file.readline()
            if tempname[-5:-1] == '.ply':
                model_list = [line.strip().replace('.ply', '') for line in file]
                model_list.insert(0, tempname[:-5])
                for j in range(len(model_list)):
                    pclTime, pclTime_color = read_points(
                        os.path.join('%s.ply' % model_list[j]), 'suncg')

                    pclTime = np.array(pclTime)
                    pclTime_color = np.array(pclTime_color)

                    pclTimeSize = [1, np.shape(pclTime)[0], np.shape(pclTime)[1]]
                    pclTime.resize(pclTimeSize)
                    pclTime_color.resize(pclTimeSize)

                    pcl = pclTime[0, :, :]
                    pcl_color = pclTime_color[0, :, :]

                    # pcl = standardize_bbox(pcl, 2048)
                    pcl = pcl[:, [2, 0, 1]]
                    # pcl[:, 0] *= -1
                    pcl[:, 2] += 0.0125

                    spotlight = np.mean(pcl[:50, :], axis=0)
                    spotlight = spotlight / np.linalg.norm(spotlight) * np.sqrt(27)
                    spotlight[2] = 3.0
                    # xml_segments = [xml_head.format(spotlight[0],spotlight[1],spotlight[2])]
                    # NOTE other dataset
                    xml_segments = [xml_head.format(-2.5,2.5,2.2)]
                    # NOTE: side view
                    # xml_segments = [xml_head.format(-3.5,0,2.2)]
                    # NOTE: front view
                    # xml_segments = [xml_head.format(0,3.5,2.2)]

                    # NOTE eye dataset
                    # xml_segments = [xml_head.format(2.7,-2.7,1.3)]
                    for i in range(pcl.shape[0]):
                        color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
                        # xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
                        xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *pcl_color[i]))
                    xml_segments.append(xml_tail.format(np.min(pcl[:, 2])-0.025))

                    xml_content = str.join('', xml_segments)

                    xmlFile = ("%s.xml" % (model_list[j]))

                    with open(xmlFile, 'w') as f:
                        f.write(xml_content)
                    f.close()

                    exrFile = ("%s.exr" % (model_list[j]))
                    if (not os.path.exists(exrFile)):
                        print(['Running Mitsuba, writing to: ', xmlFile])
                        subprocess.run([PATH_TO_MITSUBA2, xmlFile])
                    else:
                        print('skipping rendering because the EXR file already exists')

                    png = ("%s.jpg" % (model_list[j]))

                    print(['Converting EXR to JPG...'])
                    ConvertEXRToJPG(exrFile, png)
            elif tempname[-5:-1] == '.obj':
                model_list = [line.strip().replace('.obj', '') for line in file]
                model_list.insert(0, tempname[:-5])
                for j in range(len(model_list)):
                    xml_segments = [xml_head]
                    xml_segments.append(obj_mesh.format(os.path.join('%s.obj' % model_list[j])))
                    xml_segments.append(xml_tail.format(-0.025))
                    # xml_segments.append(xml_tail)

                    xml_content = str.join('', xml_segments)

                    xmlFile = ("%s.xml" % (model_list[j]))

                    with open(xmlFile, 'w') as f:
                        f.write(xml_content)
                    f.close()

                    exrFile = ("%s.exr" % (model_list[j]))
                    if (not os.path.exists(exrFile)):
                        print(['Running Mitsuba, writing to: ', xmlFile])
                        subprocess.run([PATH_TO_MITSUBA2, xmlFile])
                    else:
                        print('skipping rendering because the EXR file already exists')

                    png = ("%s.jpg" % (model_list[j]))

                    print(['Converting EXR to JPG...'])
                    ConvertEXRToJPG(exrFile, png)
    else:
        print('unsupported file format.')
        return


if __name__ == "__main__":
    main(sys.argv)
