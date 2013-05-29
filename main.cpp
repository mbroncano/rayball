#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cfloat>
#include <time.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <GLUT/glut.h>

// from https://github.com/brandonpelfrey/Fast-BVH
#include "Vector3.h"


using namespace std;

const Vector3 vec_zero = Vector3(0.f, 0.f, 0.f);

float frandom() {
	return (float)rand()/(float)RAND_MAX;
}

template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

union RGBA{
	uint32_t rgba;
	struct { uint8_t r, g, b, a; };
	uint8_t comp[4];
	
	RGBA(const Vector3& sample) {
		for (int i = 0; i < 4; i++) {
			comp[i] = min(int(sample[i] * 256), 255);
		}
	}
	
	static void srgbEncode(Vector3& c)
	{
		for (int i =0; i< 4; i++) {
		    if (c[i] <= 0.0031308f) {
		        c[i] *= 12.92f; 
		    } else {
		        c[i] *= 1.055f * powf(c[i], 0.4166667f) - 0.055f; // Inverse gamma 2.4
			}
		}
	}

	static void exposure(Vector3& c, float e) {
		for (int i =0; i< 4; i++) {
			c[i] = 1.f - expf(c[i] * e);
		}
	}
};

struct Ray {
	Vector3 origin;
	Vector3 direction;
	
	Ray(const Vector3& o, const Vector3& d) {
		origin = o;
		direction = d;
	}
	
	inline Ray reflect(const Vector3& hitPoint, const Vector3 &normal) {
		float cosI = -1.f * direction * normal;
		
		// applies the smallest offset over the hit point to be sure not to hit again the sphere
		return Ray(hitPoint + normal * (1.f + FLT_EPSILON), direction + 2.f * cosI * normal);
	}
};

struct Camera {
	Vector3 position;
	Vector3 target;
	Vector3 direction;
	
	int width, height;
	Vector3 vec_x, vec_y;

	Camera(const Vector3& p, const Vector3& t) {
		position = p;
		target = t;
		direction = normalize(t - p);
	}
	
	void setSize(int w, int h) {
		width  = w;
		height = h;

		float fov = M_PI / 180.f * 45.f;
		
		const Vector3 vec_up = Vector3(0.0f, 1.0f, 0.0f);
		vec_x = normalize(direction ^ vec_up) * (width * fov / height);
		vec_y = normalize(vec_x ^ direction) * fov;
	}
	
	Ray createRay(float x, float y) {
		float fx = x / width - 0.5f;
		float fy = y / height - 0.5f;
		
		return Ray(position, vec_x * fx + vec_y * fy + direction);
	}
	
};

struct Primitive {
	const char *name;

	Vector3 vel;

	Vector3 color;
	Vector3 emission;
	int material;

	Primitive(const char *n, const Vector3& col, const Vector3& e, int m) {
		name = n;
		color = col;
		emission = e;
		material = m;
		
		vel = vec_zero;
	}

	inline bool isLight() {
		return emission.x != 0 || emission.y != 0 || emission.x != 0;
	};

	virtual float getDistance(const Ray& r) = 0;
	virtual void getTextureCoordinates(const Vector3& point, float& u, float &v) = 0;
	virtual Vector3 getNormal(const Vector3& point) = 0 ;
	virtual Vector3 getSurfacePoint(const float u1, const float u2) = 0;
	virtual Vector3 getCenter() = 0;
};

struct Triangle : Primitive {

	Vector3 p[3]; // points
	Vector3 edge[2]; // edges
	Vector3 center;
	Vector3 normal;
	
	Triangle (const char *n, const Vector3& a, const Vector3& b,const Vector3& c, const Vector3& col, const Vector3& e, int m) : Primitive(n, col, e, m) {
		p[0] = a;
		p[1] = b;
		p[2] = c;
		
		center = (a + b + c) / 3.f;
		normal = normalize((b - a) ^ (c - a));

		edge[0] = p[1] - p[0];
		edge[1] = p[2] - p[0];
	}
	
	Vector3 getCenter() {
		return center;
	}
	
	// no phong interpolation here
	Vector3 getNormal(const Vector3& point) {
		return normal; 
	}
	
	// TODO
	Vector3 getSurfacePoint(const float u1, const float u2) {
		return vec_zero;
	}
	
	// TODO: get them from getDistance
	void getTextureCoordinates(const Vector3& point, float& u, float &v) {
		u = v = 0;
	}
	
	// Moller - Trumbore method
	// http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
	float getDistance(const Ray& r) {
		float d, u, v;
		
		Vector3 t = (r.origin - p[0]);
		Vector3 p = (r.direction ^ edge[1]);
		
		float det1 = p * edge[0];
		if (fabs(det1) < FLT_EPSILON) {
			return -1.f;
		} else {
			float u = (p * t) / det1;
			if (u < 0.f || u > 1.f)
				return -1.f;

			Vector3 q = (t ^ edge[0]);
			float v = (q * r.direction) / det1;
			if (v < 0.f || (u + v) > 1.f)
				return -1.f;

			return (q * edge[1]) / det1;
		}
	}
};

struct Square : Triangle {
	
	Square (const char *n, const Vector3& a, const Vector3& b,const Vector3& c, const Vector3& col, const Vector3& e, int m) : Triangle(n, a, b, c, col, e, m) {};
	
	// slightly changed from the triangle
	float getDistance(const Ray& r) {
		float d, u, v;

		Vector3 t = (r.origin - p[0]);
		Vector3 p = (r.direction ^ edge[1]);

		float det1 = p * edge[0];
		if (fabs(det1) < FLT_EPSILON) {
			return -1.f;
		} else {
			float u = (p * t) / det1;
			if (u < 0.f || u > 1.f)
				return -1.f;

			Vector3 q = (t ^ edge[0]);
			float v = (q * r.direction) / det1;
			if (v < 0.f || v > 1.f)
				return -1.f;

			return (q * edge[1]) / det1;
		}
	}
};

struct AABB : Primitive {

	Vector3 min;
	Vector3 max;
	Vector3 center;

	AABB(const char *n, const Vector3& a, const Vector3& b, const Vector3& col, const Vector3& e, int m) : Primitive(n, col, e, m) {
		
		for(int i =0; i < 3; i ++) {
			min[i] = std::min(a[i], b[i]);
			max[i] = std::max(a[i], b[i]);
			center[i] = (min[i] + max[i]) * 0.5f;
		}
	}
	
	Vector3 getCenter() {
		return (min + max) * 0.5;
	}
	
	Vector3 getSurfacePoint(const float u1, const float u2) {
		// TODO!!
		return getCenter();
	}

	// TO BE CHECKED!! assumes a point on the surface
	Vector3 getNormal(const Vector3& point) {
		Vector3 normal = vec_zero;
		
		for (int i = 0; i < 3; i++) {
			if (point[i] > max[i]) {
				normal[i] = 1.f;
			} else if (point[i] < min[i]) {
				normal[i] = -1.f;
			}
		}
		
		return normal;
	}
	
	// TODO!!
	void getTextureCoordinates(const Vector3& point, float& u, float &v) {
		u = v = 0.f;
	}
	
	float getDistance(const Ray& r) {
		Vector3 tmin, tmax;
		
		tmin = (min - r.origin).cdiv(r.direction);
		tmax = (max - r.origin).cdiv(r.direction);
		
		for (int i = 0; i < 3; i ++)
			if (tmin[i] > tmax[i])
				swap(tmin[i], tmax[i]);

		float dmin = tmin.x;
		float dmax = tmax.x;

		if ((dmin > tmax.y) || (tmin.y > dmax))
			return -1.f;

		dmin = std::max(dmin, tmin.y);
		dmax = std::min(dmax, tmax.y);
		
		if ((dmin > tmax.z) || (tmin.z > dmax))
			return -1.f;
		
		dmin = std::max(dmin, tmin.z);
		dmax = std::min(dmax, tmax.z);
		
		return dmin;
	}
};


struct Plane : Primitive {
	
	Vector3 point;
	Vector3 normal;
	
	Plane(const char *n, const Vector3& p, const Vector3& norm, const Vector3& col, const Vector3& e, int m) : Primitive(n, col, e, m) {
		point = p;
		normal = normalize(norm);
	}
	
	Vector3 getNormal(const Vector3& point) {
		return normal;
	}
	
	float getDistance(const Ray& r) {
		return (point - r.origin) * normal / (r.direction * normal);
	}
	
	Vector3 getCenter() {
		return point;
	}
	
	void getTextureCoordinates(const Vector3& p, float& u, float &v) {
		u = 0;
		v = 0;
	}
	
	Vector3 getSurfacePoint(const float u1, const float u2) {
		return point;
	}
	
};


struct Sphere : Primitive {	
	Vector3 center;
	float radius;

	Sphere(const char *n, const Vector3& c, float r, const Vector3& col, const Vector3& e, int m) : Primitive(n, col, e, m) {
		center = c;
		radius = r;
	}

	float getDistance(const Ray& r) {
		Vector3 v = r.origin - center;
		
		float a = r.direction * r.direction;
		float b = 2 * v * r.direction;
		float c = v * v - (radius * radius);
				
		float disc = b * b - 4 * a * c;
		if (disc < 0) {
			return -1;
		}
		
		disc = sqrtf(disc);
		float s0 = -b + disc;
		float s1 = -b - disc;
		float divis = 2 * a;
				
		if (s0 < 0) {
			return s1 / divis;
		} else if (s1 < 0) {
			return s0 / divis;
		} else if (s0 > s1) {
			return s1 / divis;
		} else {
			return s0 / divis;
		}
	}
	
	inline void getTextureCoordinates(const Vector3& point, float& u, float &v) {
		Vector3 normal = this->getNormal(point);
		
		u = atan(normal.z/normal.x) / M_PI - 0.5f;
		v = asin(normal.y) / M_PI - 0.5f;
	}
	
	inline Vector3 getNormal(const Vector3& spherePoint) {
		return normalize(spherePoint - center);
	}

	inline Vector3 getSurfacePoint(const float u1, const float u2) {
		const float zz = 1.f - 2.f * u1;
		const float r = sqrt(fabs(1.f - zz * zz));
		const float phi = 2.f * M_PI * u2;
		const float xx = r * cos(phi);
		const float yy = r * sin(phi);

		return center + Vector3(xx, yy, zz) * (radius - FLT_EPSILON);
	}
	
	inline Vector3 getCenter() {
		return center;
	}
};

ostream& operator<< (ostream& os, const Primitive& p) {
	os << "[Primitive: " << p.name << "]";
	return os;
} 

ostream& operator<< (ostream& os, const Sphere& s) {
	os << "[Sphere: " << s.name << "]";
	return os;
} 


enum {
	DIFF, SPEC, REFR, CHECKER
};

typedef vector<Primitive *> sphere_vec_t;

struct Scene {
	Camera *camera;	
	sphere_vec_t *sphere_vec;
	sphere_vec_t *light_vec;
	
	~Scene() {
		delete sphere_vec;
		delete light_vec;
		delete camera;
	}
	
	Scene() {
		
		sphere_vec = new sphere_vec_t();
		light_vec = new sphere_vec_t();

		sphere_vec->push_back(new Sphere("mirror",	Vector3(27.f, 16.5f, 47.f), 			16.5f,	Vector3(.9f, .9f, .9f),		vec_zero,			SPEC));
//		sphere_vec->push_back(new Sphere("ball",	Vector3(50.f, 16.5f, 57.f), 			16.5f,	Vector3(.25f, .75f, .25f),	vec_zero,			CHECKER));
		sphere_vec->push_back(new Sphere("glass",	Vector3(73.f, 16.5f, 78.f), 			16.5f,	Vector3(.9f, .9f, .9f),		vec_zero,			REFR));

		sphere_vec->push_back(new Sphere("light",	Vector3(50.f, 81.6f - 15.f, 81.6f), 	7.f,	vec_zero,			 Vector3(12.f, 12.f, 12.f),	DIFF));
		/*
		sphere_vec->push_back(new Sphere("light r",	Vector3(50.f, 81.6f - 15.f, 81.6f), 	7.f,	vec_zero,			 Vector3(48.f, 1.f, 1.f),	DIFF));
		sphere_vec->push_back(new Sphere("light b",	Vector3(40.f, 71.6f - 15.f, 71.6f), 	7.f,	vec_zero,			 Vector3(1.f, 1.f, 48.f),	DIFF));
		sphere_vec->push_back(new Sphere("light g",	Vector3(30.f, 61.6f - 15.f, 61.6f), 	7.f,	vec_zero,			 Vector3(1.f, 48.f, 1.f),	DIFF));
		*/
/*		sphere_vec->push_back(new Plane("bottom",	Vector3(0.f, 0.f, 0.f),		Vector3(0.f, 1.f, 0.f),		Vector3(.75f, .75f, .75f),	vec_zero,	DIFF));
		sphere_vec->push_back(new Plane("top",		Vector3(0.f, 81.6f, 0.f),	Vector3(0.f, -1.f, 0.f),	Vector3(.75f, .75f, .75f),	vec_zero,	DIFF));
		sphere_vec->push_back(new Plane("right",	Vector3(99.f, 0.f, 0.f),	Vector3(-1.f, 0.f, 0.f),	Vector3(.25f, .25f, .75f),	vec_zero,	DIFF));
		sphere_vec->push_back(new Plane("left",		Vector3(0.f, 0.f, 0.f),		Vector3(1.f, 0.f, 0.f),		Vector3(.75f, .25f, .25f),	vec_zero,	DIFF));
		sphere_vec->push_back(new Plane("back",		Vector3(0.f, 0.f, 0.f),		Vector3(0.f, 0.f, -1.f),	Vector3(.75f, .75f, .75f),	vec_zero,	DIFF));
*/
//		sphere_vec->push_back(new Plane("front",	Vector3(0.f, 0.f, 20.f), Vector3(0.f, 0.f, 1.f),	vec_zero,		vec_zero,	DIFF));

		sphere_vec->push_back(new AABB("top light",		Vector3(40.f, 80.f, 40.f),	Vector3(60.f, 85.f, 60.f),	Vector3(.25f, .75f, .75f),	Vector3(12.f, 12.f, 12.f),	DIFF));
//		sphere_vec->push_back(new AABB("water",		Vector3(0.f, 0.f, 0.f),	Vector3(100.f, 25.f, 90.f),	Vector3(.25f, .75f, .75f),	vec_zero,	REFR));
		
		sphere_vec->push_back(new Square("floor",	Vector3(0.f, 0.f, 0.f),
													Vector3(0.f, 0.f, 100.f), 
													Vector3(100.f, 0.f, 0.f),	Vector3(.75f, .75f, .75f),	vec_zero,	DIFF));

		sphere_vec->push_back(new Square("ceiling",	Vector3(0.f, 81.6f, 0.f),
													Vector3(0.f, 81.6f, 100.f), 
													Vector3(100.f, 81.6f, 0.f),	Vector3(.75f, .75f, .75f),	vec_zero,	DIFF));

		sphere_vec->push_back(new Square("back",	Vector3(0.f, 0.f, 0.f),
													Vector3(0.f, 81.6f, 0.f), 
													Vector3(100.f, 0.f, 0.f),	Vector3(.75f, .75f, .75f),	vec_zero,	DIFF));

		sphere_vec->push_back(new Square("left",	Vector3(0.f, 0.f, 0.f),
													Vector3(0.f, 81.6f, 0.f), 
													Vector3(0.f, 0.f, 100.f),	Vector3(.75f, .25f, .25f),	vec_zero,	DIFF));

		sphere_vec->push_back(new Square("right",	Vector3(100.f, 0.f, 0.f),
													Vector3(100.f, 81.6f, 0.f), 
													Vector3(100.f, 0.f, 100.f),	Vector3(.25f, .25f, .75f),	vec_zero,	DIFF));
		

		
		for (sphere_vec_t::iterator it = sphere_vec->begin(); it != sphere_vec->end(); ++it) {
			if ((*it)->isLight()) {
				light_vec->push_back(*it);
			}
		}

		camera = new Camera(Vector3(50.f, 45.f, 205.6f), Vector3(50.f, 45.f - 0.042612f, 204.6f));
	}
	
	Primitive *intersectRay(const Ray& r, float& distance) {
		distance = FLT_MAX;
		Primitive *ret = NULL;

		for (sphere_vec_t::iterator it = sphere_vec->begin(); it != sphere_vec->end(); ++it) {
			Primitive *s = *it;

			float d = s->getDistance(r);
			if (d > 0 && d < distance) {
				distance = d;
				ret = s;
			}
		}

		return ret;
	}
	
	void tick() {
		const Vector3 acc = Vector3(0.f, -0.9f, 0.f);
		const Vector3 aa = Vector3(0.f, 0.f, 0.f);
		const Vector3 bb = Vector3(100.f, 100.f, 100.f);
		
		// animate lights
		for (sphere_vec_t::iterator it = light_vec->begin(); it != light_vec->end(); ++it) {
			Sphere *light = dynamic_cast<Sphere *>(*it);
			
			// only bouce spheres
			if (light == NULL)
				continue;

			for (int i =0; i<3; i++) {
				if ((light->center[i] < (aa[i] + light->radius*2)) || (light->center[i] > (bb[i] - light->radius*2))){
					light->vel[i] *= -1.f;
					
					for (int j = 0; j< 3; j ++) {
						if (i != j)
							light->vel[j] += (frandom() * 2.f - 1.f) * 1.f;
					}
					
				} else {
					light->vel[i] += acc[i];
				}
				
				light->center[i] += light->vel[i];
			}
		}
	}
};


struct RayTracer {

	Scene *scene;
	int width;
	int height;
	int max_depth;
	int pixel_samples;
	int light_samples;
	bool soft_shadows;
	
	RGBA *fb;

	RayTracer(Scene *s, int w, int h, int md, int ps, int ls, bool ss) {
		scene = s;
		width = w;
		height = h;
		max_depth = md;
		pixel_samples = ps;
		light_samples = ls;
		soft_shadows = ss;

		fb = (RGBA *)calloc(width * height, sizeof(RGBA));
	}

	void rayTrace() {
		int sample_dir = sqrt(pixel_samples);

		scene->camera->setSize(width, height);
		
	#pragma omp parallel for schedule(dynamic,1)
		for (int y = 0; y < height; y ++) {
			for (int x = 0; x < width; x ++) {
				Vector3 sample = vec_zero;
			
				for (int sy = 0; sy < sample_dir; sy ++) {
					for (int sx = 0; sx < sample_dir; sx ++) {
						float dx = x;
						float dy = y;
					
						if (sample_dir > 1) {
							dx += float(sx) / sample_dir;
							dy += float(sy) / sample_dir;
						}
					
						Ray eyeRay = scene->camera->createRay(dx, dy);
						sample = sample + sampleRay(eyeRay, max_depth);
					}
				}	
				
				sample = sample * 1.f / (pixel_samples);

				// gamma correction, exposure
				RGBA::exposure(sample, -2.f);
				RGBA::srgbEncode(sample);

				fb[y * width + x] = RGBA(sample);
			}
		}
	}
	
	Vector3 sampleRay(Ray& ray, int depth) {

		Vector3 sample = vec_zero;

		if (depth-- == 0) {
			return sample;
		}

		float distance;
		Primitive *s = scene->intersectRay(ray, distance);
		
		if (s == NULL) {
			return sample;
		} else if (s->isLight()) {
			return s->emission;
		}

		Vector3 illumination = vec_zero;
		Vector3 hitPoint = ray.origin + distance * ray.direction;
		Vector3 normal = s->getNormal(hitPoint);
		Vector3 color = s->color;
		int material = s->material;

		switch(material) {
			case CHECKER: {
				float u,v;

				s->getTextureCoordinates(hitPoint, u, v);

				int a = int(u * 8.f) % 2;
				int b = int(v * 8.f) % 2;
				color = (a ^ b)? color * 0.6f : color;
			}
			// NOTE THE MISSING BREAK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			case DIFF: {

				// we want also to illuminate the inside of the sphere, so we
				// will invert the normal if we are inside of the sphere (i.e. the ray origin is inside it)
				// rationale: if they are in the same direction, the cos/dot is positive
				if (normal * ray.direction > 0.f) {
					normal = -1.f * normal;
				}
				illumination = sampleLights(ray, hitPoint, normal);

				// lambert model
				Vector3 ambient = Vector3(1.0f, 1.0f, 1.0f) * 0.7f;
				Vector3 intensity = ambient + illumination;
				sample = color.cmul(intensity);
			}
			break;
			case SPEC: {
				// as before, we want to reflect the interior of a sphere the same
				if (normal * ray.direction > 0.f) {
					normal = -1.f * normal;
				}

				Ray reflected = ray.reflect(hitPoint, normal);
				sample = sampleRay(reflected, depth).cmul(color);
			}
			break;
			case REFR: {
				float n1 = 1.f;  // air
				float n2 = 1.5f; // cristal

				float cosI = -1.f * ray.direction * normal;
				// if the ray and the normal are on the same direction
				// we flip the normal, invert the cosine and make the transition n2 -> n1
				if (cosI < 0.f) {
					normal = -1.f * normal;
					cosI = -cosI;
					float tmp = n1; n1 = n2; n2 = tmp;
				}
				float n = n1 / n2;

				float cosT2 = 1.0f - n * n * (1.0f - cosI * cosI);
				// total internal reflection
				if (cosT2 < 0.f) {
					Ray reflected = ray.reflect(hitPoint, normal);
					sample = sampleRay(reflected, depth);
				} else {
					float cosT = sqrtf(cosT2);

					// Fresnell's equations, per polarization (averaged)
					float perp = pow((n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT), 2.f);
					float para = pow((n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT), 2.f);
					float fres = (perp + para) / 2.f;

					Ray reflected = ray.reflect(hitPoint, normal);
					sample = fres * sampleRay(reflected, depth);

					Ray refracted = Ray(hitPoint - normal * (1.f + FLT_EPSILON), n * ray.direction + (n * cosI - cosT) * normal);
					sample = sample + (1.f - fres) * sampleRay(refracted, depth);

					sample = sample.cmul(color);
				}
			}
			break;
		}

		return sample;
	}
	
	Vector3 sampleLights(const Ray& ray, const Vector3& hitPoint, const Vector3& normal) {
		Vector3 total = vec_zero;

		for (sphere_vec_t::iterator it = scene->light_vec->begin(); it != scene->light_vec->end(); ++it) {
			Primitive *l = *it;

			Vector3 illumination = vec_zero;

			for(int j =0; j < light_samples; j ++) {
				// the default light point is the center of the sphere
				Vector3 lightPoint;

				// chooses a random point over the sphere
				if (soft_shadows) {
					lightPoint = l->getSurfacePoint(frandom(), frandom());
					// TODO: check and correct if the point is at the other side of the sphere
				// special case for 1 sample, the centre of the light (whitted)
				// for the rest, distributed over the sphere
				} else if (light_samples > 1) {
					float t = float(j) / light_samples;
					lightPoint = l->getSurfacePoint(t, t);
				} else {
					lightPoint = l->getCenter();
				}

				// creates a shadow ray, the light point should be inside of the light sphere
				// the origin is just a little bit away from the surface
				Vector3 lightVector = lightPoint - hitPoint;
				Vector3 origin = hitPoint + normal * (1.f + FLT_EPSILON);
				Ray shadowRay = Ray(origin, lightVector);

				float distance;
				Primitive *s = scene->intersectRay(shadowRay, distance);

				// the nearest intersection should be the light
				if (s == l) {
					// normalized light vector
					Vector3 normalLightVector = normalize(lightVector);

					// lambert cosine (diffuse component)
					float lambert = std::max(0.f, normal * normalLightVector);

					// phong illumination (specular component)
					float nshiny = 8.f; // the higher the smaller the spot
					/*
					Vector3 normalReflect = normalize(reflectVector(ray.direction, normal));
					float phong = powf(std::max(0.f, normalReflect * normalLightVector), nshiny);
					*/
					// blinn & torrance alternative to phong (faster to compute, similar)
					float blinn = powf(normalize(lightVector+ray.direction) * normal, nshiny * 2.f); 

					// lenght of the light vector
					float attenuation = sqrtf(lightVector * lightVector);
					Vector3 contribution = l->emission * (lambert + blinn) * 0.5f / attenuation;

					illumination = illumination + contribution;
				}
			}

				// averages illumination over the samples
			total = total + illumination / light_samples;
		};

		return total;
	}
};

#define RES_X 256
#define RES_Y RES_X
#define MAX_DEPTH 6
#define PIXEL_SAMPLES 4
#define LIGHT_SAMPLES 1
#define STOCHASTIC_LIGHT_SAMPLING false

// Render and scene
RayTracer *rayTracer;
Scene *scene;

// GLUT, OpenGL related functions
GLuint textid = 0;
char label[256];
int screen_w, screen_h;


void reshape(int width, int height) {
	screen_w = width;
	screen_h = height;
	glViewport(0, 0, width, height);
}

void display() {
	glLoadIdentity();
	glOrtho(0.f, screen_w - 1.f, 0.f, screen_h - 1.f, -1.f, 1.f);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	if (textid) {
		glBindTexture(GL_TEXTURE_2D, textid);
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex2f(screen_w - 1.0f, 0.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex2f(screen_w - 1.0f, screen_h - 1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f,  screen_h - 1.0f);
		glEnd();
	} else {
		sprintf(label, "waiting for the first frame ...");
	}
	
	glEnable(GL_BLEND);
	glBlendFunc (GL_ONE, GL_ONE);
	glColor4f(0.f, 0.f, 0.8f, 0.7f);
	glRecti(10.f, 10.f, screen_w - 10.f, 40.f);
	
	glColor3f(1.f, 1.f, 1.f);
	glRasterPos2f(15.f, 20.f);
	for (int i = 0; i < strlen (label); i++)
 		glutBitmapCharacter (GLUT_BITMAP_HELVETICA_18, label[i]);

	glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y) {
	switch(key) {
		case 27: exit(0);
		default: break;
	}
}

void idle() {	
	if (!textid) {
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1, &textid);
		printf("created texture, textid: %d\n", textid);
	}
	
	clock_t tick = clock();
	
	rayTracer->rayTrace();
	scene->tick();	
	
	float seconds = float(clock() - tick) / CLOCKS_PER_SEC;
	sprintf(label, "size: (%d, %d), frame: %0.3fs, samples: %d", RES_X, RES_Y, seconds, PIXEL_SAMPLES);	
	
	glBindTexture(GL_TEXTURE_2D, textid);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RES_X, RES_Y, 0, GL_RGBA, GL_UNSIGNED_BYTE, rayTracer->fb);
	
	glutPostRedisplay();
}

int main(int argc, char **argv) {
	
	scene = new Scene();
	rayTracer = new RayTracer(scene, RES_X, RES_X , MAX_DEPTH, PIXEL_SAMPLES, LIGHT_SAMPLES, STOCHASTIC_LIGHT_SAMPLING);
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
	glutInitWindowSize(1024, 1024);
	glutInitWindowPosition(1300, 50);
	glutCreateWindow("Rayball3");

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);

	glutMainLoop();
}
