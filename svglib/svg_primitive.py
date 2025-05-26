from __future__ import annotations
from .geom import *
import re
from typing import List, Union
from xml.dom import minidom
from .svg_path import SVGPath
from .svg_command import SVGCommandLine, SVGCommandArc, SVGCommandBezier, SVGCommandClose
import shapely
import shapely.ops
import shapely.geometry
import networkx as nx


FLOAT_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def extract_args(args):
    return list(map(float, FLOAT_RE.findall(args)))


class SVGPrimitive:
    """
    Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Basic_Shapes
    """
    def __init__(self, color="black", fill=False, dasharray=None, stroke="#000000", stroke_width=".3", opacity=1.0, id=""):
        self.color = color
        self.dasharray = dasharray
        self.stroke = stroke
        self.stroke_width = stroke_width
        self.opacity = opacity

        self.fill = fill

    def _get_fill_attr(self):
        fill_attr = f'fill="{self.color}" fill-opacity="{self.opacity}"'
        stroke_attr = f'stroke="{self.stroke}" stroke-width="{self.stroke_width}" stroke-opacity="{self.opacity}" stroke-linecap="round" stroke-linejoin="round"'
        
        if not self.fill:
            fill_attr = 'fill="none"'
        
        fill_attr += f' {stroke_attr}'
        
        if self.dasharray is not None:
            fill_attr += f' stroke-dasharray="{self.dasharray}"'
        
        return fill_attr

    @classmethod
    def from_xml(cls, x: minidom.Element):
        raise NotImplementedError

    def draw(self, viewbox=Bbox(24), *args, **kwargs):
        from .svg import SVG
        return SVG([self], viewbox=viewbox).draw(*args, **kwargs)

    def _get_viz_elements(self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=True, with_moves=True):
        return []

    def to_path(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def bbox(self):
        raise NotImplementedError

    def fill_(self, fill=True):
        self.fill = fill
        return self


class SVGEllipse(SVGPrimitive):
    def __init__(self, center: Point, radius: Radius, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.center = center
        self.radius = radius

    def __repr__(self):
        return f'SVGEllipse(c={self.center} r={self.radius})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return f'<ellipse {fill_attr} cx="{self.center.x}" cy="{self.center.y}" rx="{self.radius.x}" ry="{self.radius.y}"/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        center = Point(float(x.getAttribute("cx")), float(x.getAttribute("cy")))
        radius = Radius(float(x.getAttribute("rx")), float(x.getAttribute("ry")))
        return SVGEllipse(center, radius, fill=fill)

    def to_path(self):
        p0, p1 = self.center + self.radius.xproj(), self.center + self.radius.yproj()
        p2, p3 = self.center - self.radius.xproj(), self.center - self.radius.yproj()
        commands = [
            SVGCommandArc(p0, self.radius, Angle(0.), Flag(0.), Flag(1.), p1),
            SVGCommandArc(p1, self.radius, Angle(0.), Flag(0.), Flag(1.), p2),
            SVGCommandArc(p2, self.radius, Angle(0.), Flag(0.), Flag(1.), p3),
            SVGCommandArc(p3, self.radius, Angle(0.), Flag(0.), Flag(1.), p0),
        ]
        return SVGPath(commands, closed=True).to_group(fill=self.fill)


class SVGCircle(SVGEllipse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'SVGCircle(c={self.center} r={self.radius})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return f'<circle {fill_attr} cx="{self.center.x}" cy="{self.center.y}" r="{self.radius.x}"/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        center = Point(float(x.getAttribute("cx")), float(x.getAttribute("cy")))
        radius = Radius(float(x.getAttribute("r")))
        return SVGCircle(center, radius, fill=fill)
    
    def translate(self, vec):
        self.center += vec
        return self

    def scale(self, factor):
        self.center.scale(factor)
        return self

class SVGRectangle(SVGPrimitive):
    def __init__(self, xy: Point, wh: Size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.xy = xy
        self.wh = wh

    def __repr__(self):
        return f'SVGRectangle(xy={self.xy} wh={self.wh})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return f'<rect {fill_attr} x="{self.xy.x}" y="{self.xy.y}" width="{self.wh.x}" height="{self.wh.y}"/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        xy = Point(0.)
        if x.hasAttribute("x"):
            xy.pos[0] = float(x.getAttribute("x"))
        if x.hasAttribute("y"):
            xy.pos[1] = float(x.getAttribute("y"))
        wh = Size(float(x.getAttribute("width")), float(x.getAttribute("height")))
        return SVGRectangle(xy, wh, fill=fill)

    def to_path(self):
        p0, p1, p2, p3 = self.xy, self.xy + self.wh.xproj(), self.xy + self.wh, self.xy + self.wh.yproj()
        commands = [
            SVGCommandLine(p0, p1),
            SVGCommandLine(p1, p2),
            SVGCommandLine(p2, p3),
            SVGCommandLine(p3, p0)
        ]
        return SVGPath(commands, closed=True).to_group(fill=self.fill, color="red", stroke_width="3", dasharray="0 4 0")

    def expand(self, num):
        self.xy -= Point(num, num)
        self.wh += Size(num * 2, num * 2)
        return self


class SVGLine(SVGPrimitive):
    def __init__(self, start_pos: Point, end_pos: Point, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_pos = start_pos
        self.end_pos = end_pos

    def __repr__(self):
        return f'SVGLine(xy1={self.start_pos} xy2={self.end_pos})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return f'<line {fill_attr} x1="{self.start_pos.x}" y1="{self.start_pos.y}" x2="{self.end_pos.x}" y2="{self.end_pos.y}"/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        start_pos = Point(float(x.getAttribute("x1") or 0.), float(x.getAttribute("y1") or 0.))
        end_pos = Point(float(x.getAttribute("x2") or 0.), float(x.getAttribute("y2") or 0.))
        return SVGLine(start_pos, end_pos, fill=fill)

    def to_path(self):
        return SVGPath([SVGCommandLine(self.start_pos, self.end_pos)]).to_group(fill=self.fill)


class SVGPolyline(SVGPrimitive):
    def __init__(self, points: List[Point], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.points = points

    def __repr__(self):
        return f'SVGPolyline(points={self.points})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return '<polyline {} points="{}"/>'.format(fill_attr, ' '.join([p.to_str() for p in self.points]))

    @classmethod
    def from_xml(cls, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        args = extract_args(x.getAttribute("points"))
        assert len(args) % 2 == 0, f"Expected even number of arguments for SVGPolyline: {len(args)} given"
        points = [Point(x, args[2*i+1]) for i, x in enumerate(args[::2])]
        return cls(points, fill=fill)

    def to_path(self):
        commands = [SVGCommandLine(p1, p2) for p1, p2 in zip(self.points[:-1], self.points[1:])]
        is_closed = self.__class__.__name__ == "SVGPolygon"
        return SVGPath(commands, closed=is_closed).to_group(fill=self.fill)


class SVGPolygon(SVGPolyline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'SVGPolygon(points={self.points})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return '<polygon {} points="{}"/>'.format(fill_attr, ' '.join([p.to_str() for p in self.points]))


class SVGText(SVGPrimitive):
    """
    Represents an SVG <text> element.
    """
    def __init__(self, x: float, y: float, text_content: str,
                 font_size: Union[str, float] = "12px", font_family: str = "sans-serif",
                 text_anchor: str = "start", fill: str = "black", # This is fill color
                 stroke: str = "none", stroke_width: Union[str, float] = "0",
                 fill_flag: bool = True, # This is the boolean fill attribute for SVGPrimitive
                 *args, **kwargs):
        super().__init__(color=fill, fill=fill_flag, stroke=stroke, stroke_width=str(stroke_width), *args, **kwargs)
        self.x = float(x)
        self.y = float(y)
        self.text_content = str(text_content)
        self.font_size = str(font_size)
        self.font_family = str(font_family)
        self.text_anchor = str(text_anchor)
        # self.fill is managed by SVGPrimitive's self.color and self.fill (boolean)

    def __repr__(self):
        return f'SVGText(x={self.x}, y={self.y}, text="{self.text_content[:20]}...", font_size="{self.font_size}")'

    @classmethod
    def from_xml(cls, x: minidom.Element):
        # Common presentation attributes from SVGPrimitive
        parsed_fill_color = x.getAttribute("fill") or "black" # Default fill for text is black
        fill_bool_flag = not (x.hasAttribute("fill") and x.getAttribute("fill") == "none")
        stroke = x.getAttribute("stroke") or "none"
        stroke_width = x.getAttribute("stroke-width") or "0"
        opacity = float(x.getAttribute("opacity") or 1.0)
        # id_attr = x.getAttribute("id") or "" # Handled by SVGPrimitive?

        # Text-specific attributes
        pos_x = float(x.getAttribute("x") or 0.0)
        pos_y = float(x.getAttribute("y") or 0.0)
        font_size = x.getAttribute("font-size") or "12px"
        font_family = x.getAttribute("font-family") or "sans-serif"
        text_anchor = x.getAttribute("text-anchor") or "start"
        
        text_content = ""
        if x.firstChild and x.firstChild.nodeType == x.firstChild.TEXT_NODE:
            text_content = x.firstChild.data.strip()
        # Consider tspan elements for more complex text structures later if needed

        return cls(x=pos_x, y=pos_y, text_content=text_content,
                   font_size=font_size, font_family=font_family,
                   text_anchor=text_anchor, fill=parsed_fill_color, # fill color for text
                   stroke=stroke, stroke_width=stroke_width,
                   opacity=opacity, # Pass standard primitive attributes
                   # id=id_attr # if SVGPrimitive handles id
                   fill_flag=fill_bool_flag # Boolean for SVGPrimitive's fill attribute
                   )

    def to_str(self, *args, **kwargs):
        # Attributes from SVGPrimitive (handled by _get_fill_attr)
        # Note: self.color in SVGPrimitive is used as the fill color here.
        # self.fill (boolean) in SVGPrimitive determines if fill="none" or uses self.color
        
        # Ensure self.color is set to the text's fill color for _get_fill_attr
        # This might be redundant if __init__ correctly maps fill to self.color
        # self.color = self.text_fill_color 

        attrs = [self._get_fill_attr()] # Gets fill, stroke, opacity etc.
        attrs.append(f'x="{self.x}"')
        attrs.append(f'y="{self.y}"')
        if self.font_size:
            attrs.append(f'font-size="{self.font_size}"')
        if self.font_family:
            attrs.append(f'font-family="{self.font_family}"')
        if self.text_anchor:
            attrs.append(f'text-anchor="{self.text_anchor}"')
        
        # Filter out empty or default attributes if necessary, though explicit is often better
        
        return f'<text {" ".join(attrs)}>{self.text_content}</text>'

    def copy(self):
        # self.fill from SVGPrimitive is the boolean flag
        # self.color from SVGPrimitive is the fill color
        return SVGText(x=self.x, y=self.y, text_content=self.text_content,
                       font_size=self.font_size, font_family=self.font_family,
                       text_anchor=self.text_anchor, fill=self.color, 
                       stroke=self.stroke, stroke_width=self.stroke_width,
                       fill_flag=self.fill, # Pass the boolean fill flag
                       opacity=self.opacity)

    def to_path(self):
        # Text-to-path conversion is complex and requires font metrics.
        # For now, as per requirements:
        raise NotImplementedError("SVGText.to_path() is not implemented.")
        # Alternatively, return SVGPathGroup([]) or similar if preferred for compatibility.

    def bbox(self):
        # Accurate bbox for text is complex without rendering.
        # Returning a zero-size bbox at the (x,y) position as per requirements.
        return Bbox(self.x, self.y, self.x, self.y)

    # fill_ method is inherited from SVGPrimitive.
    # It sets self.fill (boolean) and returns self.
    # For SVGText, fill color is primarily self.color (from SVGPrimitive's perspective)
    # and self.fill (boolean) controls whether fill="none" is used.
    # The __init__ sets self.fill = True by default if a fill color is provided.


class SVGPathGroup(SVGPrimitive):
    def __init__(self, svg_paths: List[SVGPath] = None, origin=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.svg_paths = svg_paths

        if origin is None:
            origin = Point(0.)
        self.origin = origin

        self.color = kwargs.get('color', '#000000')
        self.stroke = kwargs.get('stroke', '#000000')
        self.stroke_width = kwargs.get('stroke_width', 0.0)

        self.desc = ""
        self.path_id = kwargs.get('id', "")

    # Alias
    @property
    def paths(self):
        return self.svg_paths

    @property
    def path(self):
        return self.svg_paths[0]

    def __getitem__(self, idx):
        return self.svg_paths[idx]

    def __len__(self):
        return len(self.paths)

    def total_len(self):
        return sum([len(path) for path in self.svg_paths])

    @property
    def start_pos(self):
        return self.svg_paths[0].start_pos

    @property
    def end_pos(self):
        last_path = self.svg_paths[-1]
        if last_path.closed:
            return last_path.start_pos
        return last_path.end_pos

    def set_origin(self, origin: Point):
        self.origin = origin
        if self.svg_paths:
            self.svg_paths[0].origin = origin
        self.recompute_origins()

    def append(self, path: SVGPath):
        self.svg_paths.append(path)

    def copy(self):
        return SVGPathGroup([svg_path.copy() for svg_path in self.svg_paths], self.origin.copy(),
                            color=self.color, fill=self.fill, dasharray=self.dasharray, stroke_width=self.stroke_width, opacity=self.opacity, stroke=self.stroke)

    def __repr__(self):
        return "SVGPathGroup({})".format(", ".join(svg_path.__repr__() for svg_path in self.svg_paths))

    def _get_viz_elements(self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=True, with_moves=True):
        viz_elements = []
        for svg_path in self.svg_paths:
            viz_elements.extend(svg_path._get_viz_elements(with_points, with_handles, with_bboxes, color_firstlast, with_moves))

        if with_bboxes:
            viz_elements.append(self._get_bbox_viz())

        return viz_elements

    def _get_bbox_viz(self):
        color = "red" if self.color == "black" else self.color
        bbox = self.bbox().to_rectangle(color=color)
        return bbox

    def to_path(self):
        return self

    def to_str(self, with_markers=False, coordinate_precision=0, *args, **kwargs):
        id_attr = f'id="{self.path_id}"'
        fill_attr = self._get_fill_attr()
        marker_attr = 'marker-start="url(#arrow)"' if with_markers else ''
        return '<path {} {} {} filling="{}" d="{}"></path>'.format(id_attr, fill_attr, marker_attr, self.path.filling,
                                                   " ".join(svg_path.to_str(coordinate_precision=coordinate_precision) for svg_path in self.svg_paths))

    def to_str_with_desc(self, with_markers=False, *args, **kwargs):
        id_attr = f'id="{self.path_id}"'
        fill_attr = self._get_fill_attr()
        marker_attr = 'marker-start="url(#arrow)"' if with_markers else ''
        template = '<path {} {} {} filling="{}" d="{}"><desc>{}</desc></path>'
        return template.format(id_attr, fill_attr, marker_attr, self.path.filling, " ".join(svg_path.to_str() for svg_path in self.svg_paths), self.desc)

    def to_tensor(self, PAD_VAL=-1):
        return torch.cat([p.to_tensor(PAD_VAL=PAD_VAL) for p in self.svg_paths], dim=0)

    def _apply_to_paths(self, method, *args, **kwargs):
        for path in self.svg_paths:
            getattr(path, method)(*args, **kwargs)
        return self

    def translate(self, vec):
        return self._apply_to_paths("translate", vec)

    def rotate(self, angle: Angle):
        return self._apply_to_paths("rotate", angle)

    def scale(self, factor):
        return self._apply_to_paths("scale", factor)

    def numericalize(self, n=256):
        return self._apply_to_paths("numericalize", n)

    def drop_z(self):
        return self._apply_to_paths("set_closed", False)

    def recompute_origins(self):
        origin = self.origin
        for path in self.svg_paths:
            path.origin = origin.copy()
            origin = path.end_pos
        return self

    def reorder(self):
        self._apply_to_paths("reorder")
        self.recompute_origins()
        return self

    def filter_empty(self):
        self.svg_paths = [path for path in self.svg_paths if path.path_commands]
        return self

    def canonicalize(self):
        self.svg_paths = sorted(self.svg_paths, key=lambda x: x.start_pos.tolist()[::-1])
        if not self.svg_paths[0].is_clockwise():
            self._apply_to_paths("reverse")

        self.recompute_origins()
        return self

    def reverse(self):
        self._apply_to_paths("reverse")

        self.recompute_origins()
        return self

    def duplicate_extremities(self):
        self._apply_to_paths("duplicate_extremities")
        return self

    def reverse_non_closed(self):
        self._apply_to_paths("reverse_non_closed")

        self.recompute_origins()
        return self

    def simplify(self, tolerance=0.1, epsilon=0.1, angle_threshold=179., force_smooth=False):
        self._apply_to_paths("simplify", tolerance=tolerance, epsilon=epsilon, angle_threshold=angle_threshold,
                             force_smooth=force_smooth)
        self.recompute_origins()
        return self

    def split_paths(self):
        return [SVGPathGroup([svg_path], origin=self.origin, color=self.color, fill=self.fill, dasharray=self.dasharray,
                             stroke_width=self.stroke_width, opacity=self.opacity, stroke=self.stroke)
                for svg_path in self.svg_paths]

    def split(self, n=None, max_dist=None, include_lines=True):
        return self._apply_to_paths("split", n=n, max_dist=max_dist, include_lines=include_lines)
    
    def subdivide(self, edges=None, vertices=None):
        return self._apply_to_paths("subdivide", edges=edges, vertices=vertices)

    def simplify_arcs(self):
        return self._apply_to_paths("simplify_arcs")
    
    def line_to_bezier(self):
        return self._apply_to_paths("line_to_bezier")
    
    def is_thin_line(self, width_threshold=5, aspect_ratio_threshold=5, tolerance=0.1):
        points = self.to_points()
        dummy_width = 0
        long_axis_point = None

        # Check if shape is closed
        if not np.allclose(points[0], points[-1], atol=tolerance):
            return False, dummy_width, long_axis_point
        
        # Remove duplicate end point
        points = points[:-1]
        if len(points) != 4:
            return False, dummy_width, long_axis_point
        
        # Calculate edge vectors
        edges = np.diff(points, axis=0, append=[points[0]])
        
        # Check opposite edges are parallel and equal in length
        for i in range(2):
            if not np.allclose(edges[i], -edges[i + 2], atol=tolerance):
                return False, dummy_width, long_axis_point
        
        # Check for perpendicular adjacent edges
        for i in range(4):
            if not np.isclose(np.dot(edges[i], edges[(i + 1) % 4]), 0, atol=tolerance):
                return False, dummy_width, long_axis_point
        
        # Calculate edge lengths
        lengths = np.linalg.norm(edges, axis=1)
        long_edge = np.round(np.max(lengths), 2)
        short_edge = np.round(np.min(lengths), 2)
        aspect_ratio = long_edge / (short_edge + 1e-6)

        # Determine which neighbor lies on the long axis
        if np.isclose(lengths[0], long_edge):
            long_axis_point = points[1]
        else:
            long_axis_point = points[3]

        is_thin_line = 0 < short_edge < width_threshold and aspect_ratio > aspect_ratio_threshold
        return is_thin_line, short_edge, long_axis_point

    def filter_consecutives(self):
        return self._apply_to_paths("filter_consecutives")

    def filter_duplicates(self):
        return self._apply_to_paths("filter_duplicates")

    def bbox(self):
        return union_bbox([path.bbox() for path in self.svg_paths])

    def to_shapely(self):
        return shapely.ops.unary_union([path.to_shapely() for path in self.svg_paths])

    def set_opacity(self, opacity):
        self.opacity = opacity

    def compute_filling(self):
        if self.fill:
            G = self.overlap_graph()

            root_nodes = [i for i, d in G.in_degree() if d == 0]

            for root in root_nodes:
                if not self.svg_paths[root].closed:
                    continue

                current = [(1, root)]

                while current:
                    visited = set()
                    neighbors = set()
                    for d, n in current:
                        self.svg_paths[n].set_filling(d != 0)

                        for n2 in G.neighbors(n):
                            if not n2 in visited:
                                d2 = d + (self.svg_paths[n2].is_clockwise() == self.svg_paths[n].is_clockwise()) * 2 - 1
                                visited.add(n2)
                                neighbors.add((d2, n2))

                    G.remove_nodes_from([n for d, n in current])

                    current = [(d, n) for d, n in neighbors if G.in_degree(n) == 0]

        return self

    def overlap_graph(self, threshold=0.9, draw=False):
        G = nx.DiGraph()
        shapes = [path.to_shapely() for path in self.svg_paths]

        for i, path1 in enumerate(shapes):
            G.add_node(i)

            if self.svg_paths[i].closed:
                for j, path2 in enumerate(shapes):
                    if i != j and self.svg_paths[j].closed:
                        overlap = path1.intersection(path2).area / path1.area
                        if overlap > threshold:
                            G.add_edge(j, i, weight=overlap)

        if draw:
            pos = nx.spring_layout(G)
            nx.draw_networkx(G, pos, with_labels=True)
            labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        return G

    def bbox_overlap(self, other: SVGPathGroup):
        return self.bbox().overlap(other.bbox())

    def to_points(self):
        return np.concatenate([path.to_points() for path in self.svg_paths])
