#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <set>
#include <iostream>

#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#include <emmintrin.h>
#include <pmmintrin.h>
#include <GLUT/glut.h>

using namespace std;

// from https://github.com/brandonpelfrey/Fast-BVH
// SSE Vector object
struct Vector3 {
	union __attribute__((aligned(16))) {
		struct { float x,y,z,w; };
		__m128 m128;
	};

	Vector3() { }
	Vector3(float x, float y, float z, float w=0.f) : m128(_mm_set_ps(w,z,y,x)) { }
	Vector3(const __m128& m128) : m128(m128) { }

	Vector3 operator+(const Vector3& b) const { return _mm_add_ps(m128, b.m128); }
	Vector3 operator-(const Vector3& b) const { return _mm_sub_ps(m128, b.m128); }
	Vector3 operator*(float b) const { return _mm_mul_ps(m128, _mm_set_ps(b,b,b,b)); }
	Vector3 operator/(float b) const { return _mm_div_ps(m128, _mm_set_ps(b,b,b,b)); }

	// Component-wise multiply and divide
	Vector3 cmul(const Vector3& b) const { return _mm_mul_ps(m128, b.m128); }
	Vector3 cdiv(const Vector3& b) const { return _mm_div_ps(m128, b.m128); }

	// dot (inner) product
	float operator*(const Vector3& b) const { 
		return x*b.x + y*b.y + z*b.z;
	}

	// Cross Product	
	Vector3 operator^(const Vector3& b) const {
		return _mm_sub_ps(
		_mm_mul_ps(
			_mm_shuffle_ps(m128, m128, _MM_SHUFFLE(3, 0, 2, 1)), 
			_mm_shuffle_ps(b.m128, b.m128, _MM_SHUFFLE(3, 1, 0, 2))),
		_mm_mul_ps(
			_mm_shuffle_ps(m128, m128, _MM_SHUFFLE(3, 1, 0, 2)), 
			_mm_shuffle_ps(b.m128, b.m128, _MM_SHUFFLE(3, 0, 2, 1)))
		);
	}

	Vector3 operator/(const Vector3& b) const { return _mm_div_ps(m128, b.m128); }
	
	// Handy component indexing 
	float& operator[](const unsigned int i) { return (&x)[i]; }
	const float& operator[](const unsigned int i) const { return (&x)[i]; }
};

inline Vector3 operator*(float a, const Vector3&b)  { return _mm_mul_ps(_mm_set1_ps(a), b.m128); }

// Length of a vector
inline float length(const Vector3& a) {
	return sqrtf(a*a);
}

// Make a vector unit length
inline Vector3 normalize(const Vector3& in) {
	Vector3 a = in;
	a.w = 0.f;

	__m128 D = a.m128;
	D = _mm_mul_ps(D, D);
	D = _mm_hadd_ps(D, D);
	D = _mm_hadd_ps(D, D);
 
	// 1 iteration of Newton-raphson -- Idea from Intel's Embree.
	__m128 r = _mm_rsqrt_ps(D);
	r = _mm_add_ps(
		_mm_mul_ps(_mm_set_ps(1.5f, 1.5f, 1.5f, 1.5f), r),
		_mm_mul_ps(_mm_mul_ps(_mm_mul_ps(D, _mm_set_ps(-0.5f, -0.5f, -0.5f, -0.5f)), r), _mm_mul_ps(r, r)));
 
	return _mm_mul_ps( a.m128, r );
}

const Vector3 vec_zero = Vector3(0.f, 0.f, 0.f);

float frandom() {
	return (float)rand()/(float)RAND_MAX;
}

struct Ray {
	Vector3 origin, direction;
	
	Ray(const Vector3& o, const Vector3& d) {
		origin = o;
		direction = d;
	}
};

struct Camera {
	Vector3 position, target, direction;
	Vector3 vec_x, vec_y;
	int width, height;

	Camera(const Vector3& p, const Vector3& t) {
		position = p;
		target = t;
		direction = normalize(t - p);
	}
	
	void setSize(int w, int h, float fov_angle = 45.f) {
		width  = w;
		height = h;

		float fov = M_PI / 180.f * fov_angle;
		
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

struct Texture {
	enum TextureFilter {
		None, Bilinear
	};
	
	int width, height;
	uint8_t *data;
	int scale;
	int size;
	TextureFilter filter;

	Texture(const char *path) {
		FILE *file = fopen(path, "r");
		if (!file || fscanf(file, "P6\n%d %d\n255\n", &width, &height) != 2) {
			cout << "Error opening file " << path << endl;
			exit(1);
		};
		
		size = width * height * 3;
		data = (uint8_t *)malloc(size);
		if (fread(data, 1, size, file) != size) {
			cout << "Error reading file contents" << path << endl;
			exit(2);
		}
	}
	
	Vector3 getPixelAt(const float u, const float v) {
		float x = u * width * scale;
		float y = v * height * scale;
		Vector3 ret = vec_zero;
		
		switch(filter) {
			case None: {
				int index = int(y * width + x) * 3 % size;
				for (int i = 0; i < 3; i++) {
					ret[i] = float(data[index + i]) / 255.f;
				}
			}
			break;
			case Bilinear: {
				for (int y0 = int(y); y0 < int(y) + 2; y0 ++)
					for (int x0 = int(x); x0 < int(x) + 2; x0 ++) {
						Vector3 sample = vec_zero;

						int index = ((y0 * width + x0) * 3) % size;
						for (int i = 0; i < 3; i++) {
							sample[i] = float(data[index + i]) / 255.f;
						}

						ret = ret + sample * (1.f - fabs((x0 - x0) * (y - y0))) / 4.f;
						//ret = ret + sample * (fabs((x - x0) * (y - y0)));
					}
			}
			break;
			}
		return ret;
	}	
};

struct Material {
	enum MaterialType {
		Diffuse, Specular, Glass, Light
	};
	
	Vector3 color;
	MaterialType type;
	float checker; // HACK!
	struct Texture *texture; // SUPER HACK!
	
	Material(const Vector3& c, const MaterialType t = Material::Diffuse) {
		color = c;
		type = t;
		
		checker = 0.f;
	}
	
	inline bool isLight() {
		return type == Light;
	};
	
	inline Vector3 getEmission() {
		return color * 12.f;
	}
};


struct Primitive {
	const char *name;
	Material *material;
	Vector3 vel;

	Primitive(const char *n, Material *m) {
		name = n;
		material = m;
		vel = vec_zero;
	}

	virtual float getDistance(const Ray& r) = 0;
	virtual void getTextureCoordinates(const Vector3& point, float& u, float &v) = 0;
	virtual Vector3 getNormal(const Vector3& point) = 0 ;
	virtual Vector3 getSurfacePoint(const float u1, const float u2) = 0;
	virtual Vector3 getCenter() = 0;
};

struct Triangle : Primitive {
	Vector3 point[3], edge[2];
	Vector3 center, normal;
	
	Triangle (const char *n, const Vector3& a, const Vector3& b,const Vector3& c, Material* m) : Primitive(n, m) {
		point[0] = a;
		point[1] = b;
		point[2] = c;
		
		center = (a + b + c) / 3.f;
		normal = normalize((b - a) ^ (c - a));

		edge[0] = point[1] - point[0];
		edge[1] = point[2] - point[0];
	}
	
	Vector3 getCenter() {
		return center;
	}
	
	Vector3 getNormal(const Vector3& point) {
		return normal; 
	}
	
	Vector3 getSurfacePoint(const float u, const float v) {
		return point[0] + edge[0]*u + edge[1]*v;
	}

	void getTextureCoordinates(const Vector3& point, float& u, float &v) {
		float l0 = length(edge[0]); // precalculate those
		float l1 = length(edge[1]);
		
		u = point * edge[0] / (l0 * l0);
		v = point * edge[1] / (l1 * l1);
	}
	
	// Moller - Trumbore method
	float getDistance(const Ray& r) {
		Vector3 t = (r.origin - point[0]);
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
	Square (const char *n, const Vector3& a, const Vector3& b,const Vector3& c, Material* m) : Triangle(n, a, b, c, m) {};
	
	// slightly changed from the triangle
	float getDistance(const Ray& r) {
		Vector3 t = (r.origin - point[0]);
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


struct Sphere : Primitive {	
	Vector3 center;
	float radius;

	Sphere(const char *n, const Vector3& c, const float r, Material* m) : Primitive(n, m) {
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
		
		// TODO: check if this is correct
		u = fabs(u);
		v = fabs(v);
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

struct Scene {
	Camera *camera;	
	vector<Primitive *> *spheres;
	vector<Primitive *> *lights;
	set<Material *> *materials;
	set<Texture *> *textures;
	
	~Scene() {
		delete spheres;
		delete lights;
		delete materials;
		delete textures;
		delete camera;
	}
	
	Scene() {
		spheres = new vector<Primitive *>();
		lights = new vector<Primitive *>();
		materials = new set<Material *>();
		textures = new set<Texture *>();
		
		Material *Mirror = new Material(Vector3(.9f, .9f, .9f),	Material::Specular);
		Material *Glass  = new Material(Vector3(.9f, .9f, .9f),	Material::Glass);
		Material *Red    = new Material(Vector3(.75f, .25f, .25f));
		Material *Blue   = new Material(Vector3(.25f, .25f, .75f));
		Material *Gray   = new Material(Vector3(.75f, .75f, .75f));
		Material *Light  = new Material(Vector3(.9f, .9f, .9f), Material::Light);
		Material *GLight = new Material(Vector3(.1f, .9f, .1f), Material::Light);
		Material *RLight = new Material(Vector3(.9f, .1f, .1f), Material::Light);
		Material *BLight = new Material(Vector3(.1f, .1f, .9f), Material::Light);
		Material *Green  = new Material(Vector3(.25f, .75f, .25f));
		Material *Wood   = new Material(vec_zero);
 		Green->checker = 8.f;

		materials->insert(Mirror);
		materials->insert(Glass);
		materials->insert(Red);
		materials->insert(Blue);
		materials->insert(Gray);
		materials->insert(Light);
		materials->insert(Green);
		materials->insert(GLight);
		materials->insert(BLight);
		materials->insert(RLight);
		materials->insert(Wood);
		
		Texture *texture = new Texture("wood.ppm");
		texture->scale = 2.f;
		texture->filter = Texture::Bilinear;
		Wood->texture = texture;
		
		textures->insert(texture);

		spheres->push_back(new Sphere("mirror",	Vector3(27.f, 16.5f, 47.f), 16.5f,	Mirror));
		spheres->push_back(new Sphere("wood",	Vector3(65.f, 16.5f, 40.f), 16.5f,	Wood));
		spheres->push_back(new Sphere("glass",	Vector3(73.f, 16.5f, 78.f), 16.5f,	Glass));
		spheres->push_back(new Sphere("glite",	Vector3(60.f, 70.f, 40.f),	4.f,	GLight));
		spheres->push_back(new Sphere("rlite",	Vector3(50.f, 70.f, 50.f),	4.f,	RLight));
		spheres->push_back(new Sphere("blite",	Vector3(40.f, 70.f, 60.f),	4.f,	BLight));
		
		spheres->push_back(new Square("light",	Vector3(40.f, 81.f, 40.f),	Vector3(40.f, 81.f, 60.f),	Vector3(60.f, 81.f, 40.f),	Light));
		spheres->push_back(new Square("bottom",	Vector3(0.f, 0.f, 0.f),		Vector3(0.f, 0.f, 120.f),	Vector3(100.f, 0.f, 0.f),	Gray));
		spheres->push_back(new Square("top",	Vector3(0.f, 81.6f, 0.f),	Vector3(0.f, 81.6f, 120.f), Vector3(100.f, 81.6f, 0.f),	Gray));
		spheres->push_back(new Square("back",	Vector3(0.f, 0.f, 0.f),		Vector3(0.f, 81.6f, 0.f),	Vector3(100.f, 0.f, 0.f),	Gray));
		spheres->push_back(new Square("left",	Vector3(0.f, 0.f, 0.f),		Vector3(0.f, 81.6f, 0.f),	Vector3(0.f, 0.f, 120.f),	Red));
		spheres->push_back(new Square("right",	Vector3(100.f, 0.f, 0.f),	Vector3(100.f, 81.6f, 0.f),	Vector3(100.f, 0.f, 120.f),	Blue));

		for (vector<Primitive *>::iterator it = spheres->begin(); it != spheres->end(); ++it) {
			if ((*it)->material->isLight()) {
				lights->push_back(*it);
			}
		}
		
		camera = new Camera(Vector3(50.f, 45.f, 205.6f), Vector3(50.f, 45.f - 0.042612f, 204.6f));
	}
	
	// TODO: implement kd-trees or BVH
	Primitive *intersectRay(const Ray& r, float& distance, Primitive *p = NULL) {
		distance = FLT_MAX;
		Primitive *ret = NULL;

		for (vector<Primitive *>::iterator it = spheres->begin(); it != spheres->end(); ++it) {
			Primitive *s = *it;

			if (s == p)
				continue;

			float d = s->getDistance(r);
			if (d > 0 && d < distance) {
				distance = d;
				ret = s;
			}
		}

		return ret;
	}
	
	Primitive *intersectRay(const Ray& r) {
		float d;
		return intersectRay(r, d);
	}
	
	void tick() {
		const Vector3 acc = Vector3(0.f, -0.9f, 0.f);
		const Vector3 aa = Vector3(0.f, 0.f, 0.f);
		const Vector3 bb = Vector3(100.f, 100.f, 100.f);
		
		// animate lights
		for (vector<Primitive *>::iterator it = lights->begin(); it != lights->end(); ++it) {
			Sphere *light = dynamic_cast<Sphere *>(*it);
			
			// only bouce light spheres
			if (light == NULL)
				continue;

			for (int i =0; i<3; i++) {
				light->center[i] += light->vel[i];

				if ((light->center[i] - light->radius < aa[i]) || (light->center[i] + light->radius > bb[i])){
					light->vel[i] *= -1.f;
					
					
					for (int j = 0; j< 3; j ++) {
						if (i != j)
							light->vel[j] += (frandom() -0.5f) * 0.9f;
					}
					
				} else {
					light->vel[i] += acc[i];
				}
				
			}
		}
	}
};

struct RayTracer {
	
	union RGBA{
		uint32_t rgba;
		uint8_t comp[4];

		RGBA(const Vector3& sample) {
			for (int i = 0; i < 4; i++) {
				comp[i] = min(int(sample[i] * 256), 255);
			}
		}
	};
	
	Scene *scene;
	int width, height;
	int max_depth;
	int pixel_samples;
	int light_samples;
	bool soft_shadows;
	
	RGBA *fb;
	
	~RayTracer() {
		free(fb);
	}

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
		srandom(0);
		
		int sample_dir = sqrt(pixel_samples);

		scene->camera->setSize(width, height);
#ifndef __APPLE__	
	    #pragma omp parallel for schedule(dynamic,1)
		for (int y = 0; y < height; y ++) {
#else
        dispatch_apply(height, dispatch_get_global_queue(0, 0), ^(size_t y){
#endif
			for (int x = 0; x < width; x ++) {
				Vector3 sample = vec_zero;
			
				for (int sy = 0; sy < sample_dir; sy ++) {
					for (int sx = 0; sx < sample_dir; sx ++) {
						float dx = x + float(sx) / sample_dir;
						float dy = y + float(sy) / sample_dir;
					
						Ray eyeRay = scene->camera->createRay(dx, dy);
						sample = sample + sampleRay(eyeRay, max_depth);
					}
				}	
				
				sample = sample / (pixel_samples);
				fb[y * width + x] = RGBA(sample);
			}
		}
#ifdef __APPLE__
    );
#endif
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
		}
		
		Material *m = s->material;
		Vector3 hitPoint = ray.origin + distance * ray.direction;
		Vector3 normal = s->getNormal(hitPoint);
		
		switch(m->type) {
			case Material::Light: {
				sample = m->getEmission();
			}
			break;
			case Material::Glass: {
				float n1 = 1.f;
				float n2 = 1.5f;

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
				if (cosT2 < 0.f) {
					// total internal reflection
					// applies the smallest offset over the hit point to be sure not to hit again the primitive
					Ray reflected = Ray(hitPoint + normal * (1.f + FLT_EPSILON), ray.direction + 2.f * cosI * normal);
					sample = sampleRay(reflected, depth);
				} else {
					float cosT = sqrtf(cosT2);

					// Fresnell's equations, per polarization (averaged)
					float perp = pow((n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT), 2.f);
					float para = pow((n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT), 2.f);
					float fres = (perp + para) / 2.f;

					Ray reflected = Ray(hitPoint + normal * (1.f + FLT_EPSILON), ray.direction + 2.f * cosI * normal);
					sample = fres * sampleRay(reflected, depth);

					Ray refracted = Ray(hitPoint - normal * (1.f + FLT_EPSILON), n * ray.direction + (n * cosI - cosT) * normal);
					sample = sample + (1.f - fres) * sampleRay(refracted, depth);

					sample = sample.cmul(m->color);
				}
			}
			break;
			case Material::Specular: {
				// as before, we want to reflect the interior of a sphere the same
				float cosI = normal * ray.direction;
				if (cosI < 0.f) {
					normal = -1.f * normal;
				}

				Ray reflected = Ray(hitPoint + normal * (1.f + FLT_EPSILON), ray.direction + 2.f * cosI * normal);
				sample = sampleRay(reflected, depth).cmul(m->color);
			}
			break;
			case Material::Diffuse: {
				// we want also to illuminate the inside of the sphere, so we
				if (normal * ray.direction > 0.f) {
					normal = -1.f * normal;
				}
				Vector3 illumination = vec_zero;
				
				for (vector<Primitive *>::iterator it = scene->lights->begin(); it != scene->lights->end(); ++it) {
					illumination = illumination + sampleLight(*it, ray, hitPoint, normal);
				}

				Vector3 ambient = Vector3(1.0f, 1.0f, 1.0f) * 0.7f;
				Vector3 intensity = ambient + illumination;
				
				// textured?
				if (m->texture) {
					float u, v;
					s->getTextureCoordinates(hitPoint, u, v);
					sample = m->texture->getPixelAt(u, v).cmul(intensity);
				} else {
					sample = m->color.cmul(intensity);
				}

				// Applies a simple checker texture over the previous result
				if (m->checker > 0.f) {
					float u,v;

					s->getTextureCoordinates(hitPoint, u, v);
					int scale = m->checker;

					int a = int(u * scale) % 2;
					int b = int(v * scale) % 2;
					sample = (a ^ b)? sample * 0.5f : sample;
				}
			}
			break;
		}

		return sample;
	}
	
	Vector3 sampleLight(Primitive *l, const Ray& ray, const Vector3& hitPoint, const Vector3& normal) {
		Vector3 illumination = vec_zero;

		for(int j = 0; j < light_samples; j ++) {
			// the default light point is the center of the sphere
			Vector3 lightPoint;

			// chooses a random point over the sphere
			if (soft_shadows) {
				lightPoint = l->getSurfacePoint(frandom(), frandom());
			} else if (light_samples > 1) {
				float t = float(j) / light_samples;
				lightPoint = l->getSurfacePoint(t, t);
			} else {
				// with no extra sampling, uses the center
				lightPoint = l->getCenter();
			}

			// creates a shadow ray, the light point should be inside of the light sphere
			// the origin is just a little bit away from the surface
			Ray sRay = Ray(hitPoint + normal * (1.f + FLT_EPSILON), lightPoint - hitPoint);
			Primitive *s = scene->intersectRay(sRay);

			// the nearest intersection should be the light
			if (s != l)
				continue;

			// lambert cosine (diffuse component)
			float lambert = std::max(0.f, normal * normalize(sRay.direction));

			// blinn & torrance alternative to phong (specular component, faster to compute, similar)
			float nshiny = 16.f; // the higher the smaller the spot
			float blinn = powf(normalize(sRay.direction + ray.direction) * normal, nshiny); 

			// lenght of the light vector
			float attenuation = sqrtf(sRay.direction * sRay.direction);

			illumination = illumination + (l->material->getEmission() * ((lambert + blinn) / 2.f) / attenuation);
		}

		return illumination / light_samples;
	}

};

// Render and scene
RayTracer *rayTracer;
Scene *scene;

// GLUT, OpenGL related functions
GLuint textid;
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

	glColor3f(1.f, 1.f, 1.f);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindTexture(GL_TEXTURE_2D, textid);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(screen_w - 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(screen_w - 1.0f, screen_h - 1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f,  screen_h - 1.0f);
	glEnd();

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
	clock_t tick = clock();

	rayTracer->rayTrace();
	scene->tick();	

	float seconds = float(clock() - tick) / CLOCKS_PER_SEC;
	sprintf(label, "size: (%d, %d), samples: %d, time: %0.3fs", rayTracer->width, rayTracer->height, rayTracer->pixel_samples, seconds);	

	glBindTexture(GL_TEXTURE_2D, textid);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rayTracer->width, rayTracer->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rayTracer->fb);

	glutPostRedisplay();
}

void init(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
	glutInitWindowSize(1024, 1024);
	glutInitWindowPosition(1300, 50);
	glutCreateWindow("Rayball3");

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &textid);
}
	
#define RES_X 256
#define RES_Y RES_X
#define MAX_DEPTH 6
#define PIXEL_SAMPLES 4
#define LIGHT_SAMPLES 1
#define STOCHASTIC_LIGHT_SAMPLING false

int main(int argc, char **argv) {
	scene = new Scene();
	rayTracer = new RayTracer(scene, RES_X, RES_X , MAX_DEPTH, PIXEL_SAMPLES, LIGHT_SAMPLES, STOCHASTIC_LIGHT_SAMPLING);
	
	init(argc, argv);
	glutMainLoop();
}
