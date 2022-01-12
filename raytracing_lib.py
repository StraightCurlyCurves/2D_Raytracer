'''
MIT License

2D_Raytracer: https://github.com/StraightCurlyCurves/2D_Raytracer
Copyright (c) 2021 Jan Schuessler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np

def wavelength_to_rgb(wavelength, gamma=0.8):
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A=0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B,A)

class C_ray:
    def __init__(self, x, y, lamda, n_in_wich_ray_starts, **kwargs):
        if len(kwargs)<=0:
            raise ValueError('Missing direction of the ray')
        self.x = [x]
        self.y = [y]
        self.wavelength = lamda
        self.current_n_medium = n_in_wich_ray_starts
        self.direction_angle = 0
        self.direction_vector = np.asarray([0,0])
        self.ray_is_ended = 0
        if 'angle' in kwargs:
            self.direction_angle = kwargs['angle']
            self.direction_vector = np.asarray([np.cos(self.direction_angle), np.sin(self.direction_angle)])    

    def getPos(self):
        return np.asarray([self.x[-1], self.y[-1]])

    def setPos(self, pos):
        self.x.append(pos[0])
        self.y.append(pos[1])

    def setAngle(self, angle):
        self.direction_angle = angle
        self.direction_vector = np.asarray([np.cos(self.direction_angle), np.sin(self.direction_angle)]) 

# class ray3D: # TODO
#     def __init__(self, x, y, z=0, lamda, direction):
#         self.x = [x]
#         self.y = [y]
#         self.z = [z]
#         self.wavelength = lamda
#         self.direction_vector = direction

#     def getPos(self):
#         return np.asarray([self.x[-1], self.y[-1], self.z[-1]])

#     def setPos(self, pos):
#         self.x.append(pos[0])
#         self.y.append(pos[1])


# class circle:
#     def __init__(self, x, y, radius):
#         self.x = x
#         self.y = y
#         self.r = radius

#     def getPos(self):
#         return np.asarray([self.x, self.y])

class C_circle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.r = radius

    def getPos(self):
        return np.asarray([self.x, self.y])


class C_lens:
    def __init__(self, x, r1, r2, d, material, max_height=5):
        self.start_pos = x
        self.r1 = r1 + 1000*(r1 == 0)
        self.r2 = r2 - 1000*(r2 == 0)
        self.r1_pos = x + self.r1
        self.r2_pos = x + d + self.r2 
        self.d = d
        self.lateral_surface = 1
        self.height = self.calc_height(max_height)
        self.material = material 

    def get_all_radii(self):
        return np.asarray([self.r1, self.r2])

    def get_all_x_pos(self):
        return np.asarray([self.r1_pos, self.r2_pos])     

    def plot(self, ax):
        # surface 1:
        angle = np.arcsin(self.height/np.abs(self.r1))
        angle_data = np.linspace(-angle+np.pi,angle+np.pi,1000)
        x_data = self.r1*np.cos(angle_data)+self.r1_pos
        y_data = self.r1*np.sin(angle_data)
        mask = y_data < self.height
        mask = np.logical_and(mask, y_data > -self.height)
        x_s1 = x_data[mask]
        y_s1 = y_data[mask]
        

        # surface 2:
        angle = np.arcsin(self.height/np.abs(self.r2))
        angle_data = np.linspace(-angle+np.pi,angle+np.pi,1000)
        x_data = self.r2*np.cos(angle_data)+self.r2_pos
        y_data = self.r2*np.sin(angle_data)
        mask = y_data < self.height
        mask = np.logical_and(mask, y_data > -self.height)
        x_s2 = x_data[mask]
        y_s2 = y_data[mask]

        # lateral surface
        if self.lateral_surface:
            ls_x = [x_s1[0], x_s2[0]]
            ls_y = np.asarray([self.height, self.height])

        # plot
        color = 'white'
        ax.plot(x_s1,y_s1,color)
        ax.plot(x_s2,y_s2,color)
        if self.lateral_surface:
            ax.plot(ls_x,ls_y,color, alpha=0.3)
            ax.plot(ls_x,-ls_y,color, alpha=0.3)

        # filling
        # y_s1 = y_s1[(y_s1 > 0)]
        # y_s2 = y_s2[(y_s2 > 0)]
        # y = y_s1 + y_s2
        # for i in range(len(y)-1):
        #     y_new = np.linspace(y[i], 0, 2)
        #     ax.fill_betweenx(y_new, x_data[i], x_data[i+1], facecolor='lightblue', alpha=0.5)


    def calc_height(self, max_height): # https://mathworld.wolfram.com/Circle-CircleIntersection.html
        intersect, x, y, _, _ = get_intersections(self.r1_pos, 0, np.abs(self.r1), self.r2_pos, 0, np.abs(self.r2))
        y = np.abs(y)
        max_height = np.abs(max_height)
        # print("Circle intersection at:", x, y)
        small_r = min(np.abs(self.r1),np.abs(self.r2))

        # if no intersection:
        if not intersect:
            if small_r < max_height:
                return small_r
            else:
                return max_height

        # if bi-concave
        if self.r1 < 0 and self.r2 > 0:
            if small_r < max_height:
                return small_r
            else:
                return max_height

        # if bi-convex
        if self.r1 > 0 and self.r2 < 0:
            if x < self.r1_pos and x > self.r2_pos:
                if y < max_height:
                    return y
                else:
                    return max_height
            else:
                if small_r < max_height:
                    return small_r
                else:
                    return max_height

        # if convex-concave
        if self.r1 > 0 and self.r2 > 0:
            if x < self.r1_pos and x < self.r2_pos:
                if y < max_height:
                    return y
                else:
                    return max_height
            else:
                if small_r < max_height:
                    return small_r
                else:
                    return max_height

        # if concave-convex
        if self.r1 < 0 and self.r2 < 0:
            if x > self.r1_pos and x > self.r2_pos:
                if y < max_height:
                    return y
                else:
                    return max_height
            else:
                if small_r < max_height:
                    return small_r
                else:
                    return max_height

        print("INTERSECTION ERROR")

class C_lens_group:
    def __init__(self, start_pos, r1, r2, d, material, max_height):
        self.lenses = []
        self.lenses.append(C_lens(start_pos, r1, r2, d, material, max_height))
        self.group_start_pos = start_pos
        self.group_end_pos = start_pos + d

    def add_lens(self, d_to_previous_lens, r1, r2, d, material, max_height):
        start_pos = self.group_end_pos + d_to_previous_lens
        self.lenses.append(C_lens(start_pos, r1, r2, d, material, max_height))
        self.group_end_pos = start_pos + d
        



def dot(a,b):
    return a[0]*b[0] + a[1]*b[1]

def hit_circle(circle, ray, isLens=False):
    cTr = ray.getPos() - circle.getPos() #vector circle center to ray origin

    # solve quadratic equation for t in: (P(t) - Circle)^2 = r^2
    '''a = dot(ray.direction_vector, ray.direction_vector)
    b = 2 * dot(cTr, ray.direction_vector)
    c = dot(cTr, cTr) - circle.r**2
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return -1
    else:
        return (-b + np.sqrt(discriminant) ) / (2.0*a)'''

    # optimized 'mitternachtsformel':
    a = dot(ray.direction_vector, ray.direction_vector)
    half_b = dot(cTr, ray.direction_vector)
    c = dot(cTr, cTr) - circle.r**2
    discriminant = half_b**2 - a*c

    if discriminant <= 0:
        return -1
    else:
        if isLens:
            if circle.r > 0:
                return (-half_b - np.sqrt(discriminant)) / a
            else:
                return (-half_b + np.sqrt(discriminant)) / a
        else:
            return (-half_b + np.sqrt(discriminant)) / a

def get_intersections(x0, y0, r0, x1, y1, r1):
    '''circle 1: (x0, y0), radius r0
    circle 2: (x1, y1), radius r1'''

    d=np.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # non intersecting
    if d > r0 + r1 :
        return (0,0,0,0,0)
    # One circle within other
    if d < abs(r0-r1):
        return (0,0,0,0,0)
    # coincident circles
    if d == 0 and r0 == r1:
        return (0,0,0,0,0)
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=np.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return (1, x3, y3, x4, y4)

def refr_index(lamda, material):
    lamda = lamda*1e6
    return np.sqrt((material[0] * lamda**2)/(lamda**2 - material[1]) + (material[2] * lamda**2)/(lamda**2 - material[3]) + (material[4] * lamda**2)/(lamda**2 - material[5]) + 1)

def raytracing(sensor_position, lenses, rays, n_environment, continue_ray_whithout_intersection=True):
    '''
    Parameters
    ----------
    ``sensor_position: float`` Position of the virtual sensor.

    ``lenses: list(C_lens)`` A list with lens objects from class type C_lens. ``C_lens_group.lenses`` contains a list with lenses.

    ``rays: list(C_ray)`` A list with ray objects from class type C_ray.

    ``n_environment: float | list`` Float for a simple refractive index, list for sellmeier equation

    ``continue_ray_whithout_intersection: bool`` Specifies wether the ray continues or not if it won't hit the last surface.

    Returns
    -------
    :``tuple(rays, intersections)`` Returns the ray objects, each containig its trace, and a list of intersections.
    '''
                


    
    
    # create container for all intersection points
    intersections = []
    intersections.clear()

    # alternating colors with wich the intersection points get colorized
    colors = ['red', 'green', 'blue']
    color = -1 # counter in for loop will increment it to 0 the first time

    # air_between:  If the radius and position of two surfaces from two lenses are the same,
    #               this flag will be set to zero (False) to skip the angle correction of the ray
    air_between = True

    # print_once = True # flag

    for index, lens in enumerate(lenses): # NESTED LOOP LEVEL 0
        try:
            # air between this and the next lens?
            if lens.get_all_x_pos()[1] == lenses[index+1].get_all_x_pos()[0] and lens.get_all_radii()[1] == lenses[index+1].get_all_radii()[0]:
                air_between = False
            else:
                air_between = True
        except IndexError:
            air_between = True
            pass
        
        for surface in range(2): # NESTED LOOP LEVEL 1
            color += 1
            if color > 2:
                color = 0
            
            # raytracing starts here
            for i in range(int(len(rays))): # NESTED LOOP LEVEL 3
                # if a ray has the flag 'rays_is_ended' or is already drawn to senor, skip this ray
                if rays[i].ray_is_ended:                 
                    continue

                # calculate intersection on the circle of ray i
                c = C_circle(lens.get_all_x_pos()[surface], 0, lens.get_all_radii()[surface])             
                t = hit_circle(c, rays[i], isLens=True)
                P = rays[i].getPos() + t*rays[i].direction_vector                

                # if there is no valid intersection between ray and surface, 
                if t == -1 or np.abs(P[1]) > lens.height:
                    if not continue_ray_whithout_intersection:
                        rays[i].ray_is_ended = True

                    # draw ray to sensor if the last surface was calculated and the ray has at least one intersection
                    if index == len(lenses)-1 and surface == 1 and len(rays[i].x) > 1 and continue_ray_whithout_intersection:
                        x = sensor_position
                        y = rays[i].getPos()[1] + (sensor_position - rays[i].getPos()[0]) * np.tan(rays[i].direction_angle)
                        P = np.asarray([x, y])
                        rays[i].setPos(P)
                    # elif print_once:
                    #     print("No Cirlce Intersection") # this print is quite annoying for the interactive plot and therefor commented
                    #     print_once = 0
                    continue          

                # update ray and intersections
                rays[i].setPos(P)
                intersections.append([P[0], P[1],'o',colors[color]])            
                    
                # normal vector of the surface at point P. Normal vector is pointing to the right
                if c.r < 0:
                    normal = (P - c.getPos())
                else:
                    normal = (c.getPos() - P)
                
                # calculate angle between ray and normal vector
                dot_product = dot(rays[i].direction_vector, normal)
                norm_product = (np.linalg.norm(rays[i].direction_vector)*np.linalg.norm(normal))
                cos_delta = round(dot_product/norm_product,15)
                delta = np.arccos(cos_delta)

                # get n1 and n2 
                n_M = refr_index(rays[i].wavelength, lens.material)
                current_n = rays[i].current_n_medium

                # calculate refraction: epsilon is the angle between the refracted ray and the normal vector
                if surface == 0:
                    sin_epsilon = (current_n * np.sin(delta))/(n_M)
                    rays[i].current_n_medium = n_M                
                if surface == 1:
                    if air_between:
                        sin_epsilon = (n_M * np.sin(delta))/(n_environment)
                        rays[i].current_n_medium = n_environment
                    else:
                        n_next_M = refr_index(rays[i].wavelength, lenses[index+1].material)
                        sin_epsilon = (n_M * np.sin(delta))/(n_next_M)
                        rays[i].current_n_medium = n_next_M

                # calculate the global angle of the refracted ray, if not total reflected. Otherways end the ray.
                if sin_epsilon > 1 or sin_epsilon < -1: # total reflection
                    rays[i].ray_is_ended = True
                else:
                    epsilon = np.arcsin(sin_epsilon)
                    normal_angle = np.arctan2(normal[1], normal[0])
                    if normal_angle >= rays[i].direction_angle:
                        theta = normal_angle - epsilon
                    else:
                        theta = normal_angle + epsilon      
                    rays[i].setAngle(theta)
                
                # connect ray to sensor when it hitted the last surface and the ray is not ended
                if index == len(lenses)-1 and surface == 1 and not rays[i].ray_is_ended:
                    x = sensor_position
                    y = rays[i].getPos()[1] + (sensor_position - rays[i].getPos()[0]) * np.tan(rays[i].direction_angle)
                    P = np.asarray([x, y])
                    rays[i].setPos(P)         

    return rays, intersections