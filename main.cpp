#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cfloat>
#include <time.h>
#include <omp.h>
#include <algorithm>
#include <GLUT/glut.h>

// from https://github.com/brandonpelfrey/Fast-BVH
#include "Vector3.h"


const Vector3 vec_zero = Vector3(0.f, 0.f, 0.f);

union RGBA{
	uint32_t rgba;
	struct { uint8_t r, g, b, a; };
	uint8_t comp[4];
	
	RGBA(const Vector3& sample) {
		for (int i = 0; i < 4; i++) {
			comp[i] = std::min(int(sample[i] * 256), 255);
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

	Camera(const Vector3& p, const Vector3& t, int w, int h) {
		position = p;
		target = t;
		width  = w;
		height = h;

		direction = normalize(t - p);
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
	float radius;
	Vector3 center;
	Vector3 color;
	Vector3 emission;
	const char *name;
	int material;

	Primitive(const char *n) { name = n; }
	float Intersect(Ray r) {};
	bool isLight() {};
};

struct Sphere : Primitive {	

	Sphere(const char *n, Vector3 c, float r, Vector3 col, Vector3 e, int m) : Primitive(n) {
		name = n;
		center = c;
		radius = r;
		color = col;
		emission = e;
		material = m;
	}

	float Intersect(Ray r) {
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
	
	inline void getTextUV(const Vector3& point, float& u, float &v) {
		Vector3 normal = this->Normal(point);
		
		u = atan(normal.z/normal.x) / M_PI - 0.5f;
		v = asin(normal.y) / M_PI - 0.5f;
	}
	
	inline Vector3 Normal(const Vector3& spherePoint) {
		return normalize(spherePoint - center);
	}

	inline bool isLight() {
		return !(emission.x == 0 && emission.y == 0 && emission.x == 0);
	};
};

void getNearestSphere(const Ray& r, Sphere *spheres, int numspheres, int& index, float& distance) {
	distance = FLT_MAX;
	index = -1;

	for (int i = 0; i< numspheres; i++) {
		Sphere *s = (spheres + i);
		
		float d = s->Intersect(r);
		if (d > 0 && d < distance) {
			index = i;
			distance = d;
		}
	}
}

enum {
	DIFF, SPEC, REFR, CHECKER
};

float frandom() {
	return (float)rand()/(float)RAND_MAX;
}

inline Vector3 uniformSampleSphere(const float u1, const float u2) {
	const float zz = 1.f - 2.f * u1;
	const float r = sqrt(fabs(1.f - zz * zz));
	const float phi = 2.f * M_PI * u2;
	const float xx = r * cos(phi);
	const float yy = r * sin(phi);

	return Vector3(xx, yy, zz);
}


Vector3 sampleLights(const Ray& ray, const Vector3& hitPoint, const Vector3& normal, Sphere *spheres, int numspheres, int numsamples, bool stochastic) {
	Vector3 total = vec_zero;
	
	for (int i = 0; i< numspheres; i++) {
		Sphere *l = (spheres + i);
		
		if (l->isLight()) {
			Vector3 illumination = vec_zero;
			
			for(int j =0; j < numsamples; j ++) {
				// the default light point is the center of the sphere
				Vector3 spherePoint = vec_zero;
			
				// chooses a random point over the sphere
				if (stochastic) {
					spherePoint = uniformSampleSphere(frandom(), frandom());
					// TODO: check and correct if the point is at the other side of the sphere
				// special case for 1 sample, the centre of the light (whitted)
				// for the rest, distributed over the sphere
				} else if (numsamples > 1) {
					float t = float(j) / numsamples;
					spherePoint = uniformSampleSphere(t, t);
				}
				
				// creates a shadow ray, the light point should be inside of the light sphere
				// the origin is just a little bit away from the surface
				Vector3 lightPoint = l->center + spherePoint * (l->radius - FLT_EPSILON);
				Vector3 lightVector = lightPoint - hitPoint;
				Vector3 origin = hitPoint + normal * (1.f + FLT_EPSILON);
				Ray shadowRay = Ray(origin, lightVector);

				float distance;
				int index;
				getNearestSphere(shadowRay, spheres, numspheres, index, distance);
	
				// the nearest intersection should be the light
				if (index == i) {
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
			total = total + illumination / numsamples;
		}
	};
	
	return total;
}

Vector3 sampleRay(Ray& ray, Sphere *spheres, int numspheres, int depth, int light_samples, int stochastic) {
	
	Vector3 sample = vec_zero;

	if (depth-- == 0) {
		return sample;
	}
	
	float distance;
	int index;
	getNearestSphere(ray, spheres, numspheres, index, distance);
	if (index == -1) {
		return sample;
	}
	
	Sphere *s = (spheres + index);
	if (s->isLight()) {
		return s->emission;
	}

	Vector3 illumination = vec_zero;
	Vector3 hitPoint = ray.origin + distance * ray.direction;
	Vector3 normal = s->Normal(hitPoint);
	Vector3 color = s->color;
	int material = s->material;
	
	switch(material) {
		case CHECKER: {
			float u,v;
			
			s->getTextUV(hitPoint, u, v);
			
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
			illumination = sampleLights(ray, hitPoint, normal, spheres, numspheres, light_samples, stochastic);

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
			sample = sampleRay(reflected, spheres, numspheres, depth, light_samples, stochastic).cmul(color);
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
				sample = sampleRay(reflected, spheres, numspheres, depth, light_samples, stochastic);
			} else {
				float cosT = sqrtf(cosT2);

				// Fresnell's equations, per polarization (averaged)
				float perp = pow((n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT), 2.f);
				float para = pow((n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT), 2.f);
				float fres = (perp + para) / 2.f;

				Ray reflected = ray.reflect(hitPoint, normal);
				sample = fres * sampleRay(reflected, spheres, numspheres, depth, light_samples, stochastic);
				
				Ray refracted = Ray(hitPoint - normal * (1.f + FLT_EPSILON), n * ray.direction + (n * cosI - cosT) * normal);
				sample = sample + (1.f - fres) * sampleRay(refracted, spheres, numspheres, depth, light_samples, stochastic);
				
				sample = sample.cmul(color);
			}
		}
		break;
	}
		
	return sample;
}

struct Scene {
	Sphere *spheres;
	int numspheres;
	Camera *camera;
	
	Scene(int width, int height) {
		static Camera cornellCamera = Camera(Vector3(50.f, 45.f, 205.6f), Vector3(50.f, 45.f - 0.042612f, 204.6f), width, height);		
		static Sphere cornellSpheres[] = {
			Sphere("left",		Vector3(1e4f + 1.f, 40.8f, 81.6f), 		1e4f,	Vector3(.75f, .25f, .25f),	vec_zero,	DIFF),
			Sphere("right",		Vector3(-1e4f + 99.f, 40.8f, 81.6f),	1e4f,	Vector3(.25f, .25f, .75f),	vec_zero,	DIFF),
			Sphere("back",		Vector3(50.f, 40.8f, 1e4f), 			1e4f,	Vector3(.75f, .75f, .75f),	vec_zero,	DIFF),
			Sphere("front",		Vector3(50.f, 40.8f, -1e4f), 			1e4f,	vec_zero,					vec_zero,	DIFF),
			Sphere("bottom",	Vector3(50.f, 1e4f, 81.6f), 			1e4f,	Vector3(.75f, .75f, .75f),	vec_zero,	DIFF),
			Sphere("top",		Vector3(50.f, -1e4f + 81.6f, 81.6f),	 1e4f,	Vector3(.75f, .75f, .75f),	vec_zero,	DIFF),
			Sphere("mirror",	Vector3(27.f, 16.5f, 47.f), 			16.5f,	Vector3(.9f, .9f, .9f),		vec_zero,	SPEC),
			Sphere("ball",		Vector3(50.f, 16.5f, 57.f), 			16.5f,	Vector3(.25f, .75f, .25f),	vec_zero,	CHECKER),
			Sphere("glass",		Vector3(73.f, 16.5f, 78.f), 			16.5f,	Vector3(.9f, .9f, .9f),		vec_zero,	REFR),
			Sphere("light",		Vector3(50.f, 81.6f - 15.f, 81.6f), 	7.f,	vec_zero,		 			Vector3(12.f, 12.f, 12.f),	DIFF),
		};
		
		camera = &cornellCamera;
		spheres = cornellSpheres;
		numspheres = 10;//sizeof(spheres) / sizeof(spheres[0]);
	}
};

#define RES_X 256
#define RES_Y RES_X
#define MAX_DEPTH 6
#define PIXEL_SAMPLES 1
#define LIGHT_SAMPLES 1
#define STOCHASTIC_PIXEL_SAMPLING false
#define STOCHASTIC_LIGHT_SAMPLING false

void *rayTrace(int width, int height) {
	
//	srand((unsigned)time(0));
	static Scene scene = Scene(width, height);

	RGBA *fb = (RGBA *)calloc(width * height, sizeof(RGBA)); // includes RGBA, clears memory
	int sample_dir = sqrt(PIXEL_SAMPLES);

	// bounce the ligth!
	Sphere *light = &scene.spheres[scene.numspheres - 1];
	static float vel_y = 0.f;
	const float acce_y = 0.8f;
	if (light->center.y < light->radius + vel_y) {
		vel_y *= -1.f;
	} else {
		vel_y += acce_y;
	}
	light->center.y -= vel_y;

	static float vel_x = 2.f;
	if ((light->center.x < light->radius) || (light->center.x > 100.f - light->radius)) {
		vel_x *= -1.f;
	}
	light->center.x -= vel_x;


#pragma omp parallel for schedule(dynamic,1)
	for (int y = 0; y < height; y ++) {
		for (int x = 0; x < width; x ++) {

			Vector3 sample = vec_zero;
			
			// antialias
			for (int sy = 0; sy < sample_dir; sy ++) {
				for (int sx = 0; sx < sample_dir; sx ++) {
					float dx = x;
					float dy = y;
					
					if (sample_dir > 1) {
						if (STOCHASTIC_PIXEL_SAMPLING != 0){
							dx += (frandom() - 0.5f);
							dy += (frandom() - 0.5f);
						} else {
							dx += float(sx) / sample_dir;
							dy += float(sy) / sample_dir;
						}
					}
					
					Ray eyeRay = scene.camera->createRay(dx, dy);
					sample = sample + sampleRay(eyeRay, scene.spheres, scene.numspheres, MAX_DEPTH, LIGHT_SAMPLES, STOCHASTIC_LIGHT_SAMPLING);
				}
			}	
			//#pragma omp critical
			{
				sample = sample * 1.f / (PIXEL_SAMPLES);

				// gamma correction, exposure
				RGBA::exposure(sample, -2.f);
				RGBA::srgbEncode(sample);

				fb[y * width + x] = RGBA(sample);
			}
		}
	}
	
	// tone mapping

	return fb;	
}

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
	void *data = rayTrace(RES_X, RES_Y);
	float seconds = float(clock() - tick) / CLOCKS_PER_SEC;
	sprintf(label, "size: (%d, %d), frame: %0.3fs, samples: %d", RES_X, RES_Y, seconds, PIXEL_SAMPLES);	
	
	glBindTexture(GL_TEXTURE_2D, textid);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RES_X, RES_Y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	free(data);	
	
	glutPostRedisplay();
}

int main(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
	glutInitWindowSize(1024, 1024);
	glutInitWindowPosition(50, 50);
	glutCreateWindow("Rayball3");

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);

	glutMainLoop();
}
