#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <iostream>
#include "shader.h"
#include "shaderprogram.h"

#include <vector>
 
/*=================================================================================================
	DOMAIN
=================================================================================================*/

// Window dimensions
const int InitWindowWidth  = 800;
const int InitWindowHeight = 800;
int WindowWidth  = InitWindowWidth;
int WindowHeight = InitWindowHeight;

// Last mouse cursor position
int LastMousePosX = 0;
int LastMousePosY = 0;

// Arrays that track which keys are currently pressed
bool key_states[256];
bool key_special_states[256];
bool mouse_states[8];

// Other parameters
bool draw_wireframe = false;

float maxDistance = 100;
int maxDepth = 3;

/*=================================================================================================
	SHADERS & TRANSFORMATIONS
=================================================================================================*/

ShaderProgram PassthroughShader;
ShaderProgram PerspectiveShader;

glm::mat4 PerspProjectionMatrix( 1.0f );
glm::mat4 PerspViewMatrix( 1.0f );
glm::mat4 PerspModelMatrix( 1.0f );

float perspZoom = 1.0f, perspSensitivity = 0.35f;
float perspRotationX = 0.0f, perspRotationY = 0.0f;

/*=================================================================================================
	OBJECTS
=================================================================================================*/

//VAO -> the object "as a whole", the collection of buffers that make up its data
//VBOs -> the individual buffers/arrays with data, for ex: one for coordinates, one for color, etc.

GLuint axis_VAO;
GLuint axis_VBO[2];

float axis_vertices[] = {
	//x axis
	-1.0f,  0.0f,  0.0f, 1.0f,
	1.0f,  0.0f,  0.0f, 1.0f,
	//y axis
	0.0f, -1.0f,  0.0f, 1.0f,
	0.0f,  1.0f,  0.0f, 1.0f,
	//z axis
	0.0f,  0.0f, -1.0f, 1.0f,
	0.0f,  0.0f,  1.0f, 1.0f
};

float axis_colors[] = {
	//x axis
	1.0f, 0.0f, 0.0f, 1.0f,//red
	1.0f, 0.0f, 0.0f, 1.0f,
	//y axis
	0.0f, 1.0f, 0.0f, 1.0f,//green
	0.0f, 1.0f, 0.0f, 1.0f,
	//z axis
	0.0f, 0.0f, 1.0f, 1.0f,//blue
	0.0f, 0.0f, 1.0f, 1.0f
};

//shapes initialization
GLuint shapes_VAO;
GLuint shapes_VBO[3];
std::vector<glm::vec4> shapes_vertices;
std::vector<glm::vec4> shapes_colors;
std::vector<glm::vec4> shapes_normal;

//plane
GLuint plane_VAO;
GLuint plane_VBO[2];
std::vector<glm::vec4> plane_vertices;
std::vector<glm::vec4> plane_colors;

//normal line
GLuint norm_VAO;
GLuint norm_VBO[2];
std::vector<glm::vec4> norm_vertices;
std::vector<glm::vec4> norm_colors;
/*=================================================================================================
	HELPER FUNCTIONS
=================================================================================================*/

void window_to_scene( int wx, int wy, float& sx, float& sy )
{
	sx = ( 2.0f * (float)wx / WindowWidth ) - 1.0f;
	sy = 1.0f - ( 2.0f * (float)wy / WindowHeight );
}

/*=================================================================================================
	SHADERS
=================================================================================================*/
//eye position
glm::vec3 eyePos(0.0f, 1.0f, 2.0f);
glm::vec3 centerPos(eyePos.x, eyePos.y, 0.0f);
glm::vec3 lightPos(-0.5f, 1.5f, -1.0f);
void CreateTransformationMatrices( void )
{
	// PROJECTION MATRIX
	PerspProjectionMatrix = glm::perspective<float>( glm::radians( 60.0f ), (float)WindowWidth / (float)WindowHeight, 0.01f, 1000.0f );

	// VIEW MATRIX

	glm::vec3 eye   (eyePos );
	glm::vec3 center(centerPos );
	glm::vec3 up    ( 0.0, 1.0, 0.0 );

	PerspViewMatrix = glm::lookAt( eye, center, up );

	// MODEL MATRIX
	PerspModelMatrix = glm::mat4( 1.0 );
	PerspModelMatrix = glm::rotate( PerspModelMatrix, glm::radians( perspRotationX ), glm::vec3( 1.0, 0.0, 0.0 ) );
	PerspModelMatrix = glm::rotate( PerspModelMatrix, glm::radians( perspRotationY ), glm::vec3( 0.0, 1.0, 0.0 ) );
	PerspModelMatrix = glm::scale( PerspModelMatrix, glm::vec3( perspZoom ) );
}

glm::vec4 shade(glm::vec4 vert_Pos, glm::vec4 vert_Color, glm::vec4 vert_Normal, bool isShadow) {
	glm::vec4 la = glm::vec4(0.5, 0.5, 0.5, 1.0);
	glm::vec4 ld = glm::vec4(0.7, 0.7, 0.7, 1.0);
	glm::vec4 ls = glm::vec4(1.0, 1.0, 1.0, 1.0);
	glm::vec4 ka = vert_Color;
	glm::vec4 kd = vert_Color;
	glm::vec4 ks = glm::vec4(1.0, 1.0, 1.0, 1.0);

	float shininess = 32.0f;

	glm::mat4 transf = PerspViewMatrix * PerspModelMatrix;

	glm::vec3 FragPos = glm::vec3(transf * vert_Pos);
	glm::vec3 FragNorm = glm::mat3(transpose(inverse(transf))) * glm::normalize(glm::vec3(vert_Normal.x, vert_Normal.y, vert_Normal.z));
	glm::vec3 LightPos = glm::vec4(lightPos, 1.0f);



	glm::vec3 N = normalize(FragNorm); // vertex normal
	glm::vec3 L = normalize(LightPos - FragPos); // light direction
	glm::vec3 R = normalize(glm::reflect(-L, N)); // reflected ray
	glm::vec3 V = normalize(glm::vec3(eyePos)); // view direction

	float dotLN = glm::dot(L, N);
	glm::vec4 amb = ka * la;
	glm::vec4 dif = kd * ld * glm::max(dotLN, 0.0f);
	glm::vec4 spe = ks * ls * glm::pow(glm::max(glm::dot(R, V), 0.0f), shininess) * glm::max(dotLN, 0.0f);
	if (isShadow) {
		return amb;
	}
	return amb + dif + spe;
}
bool rayTriangleIntersect(glm::vec3& orig, glm::vec3& dir, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float& t) {
	//compute the triangle's normal
	glm::vec3 v1v0 = v1 - v0;
	glm::vec3 v2v0 = v2 - v0;

	glm::vec3 N = glm::cross(v1v0, v2v0);
	N = glm::normalize(N);
	//find P, the intersection point
	//check if triangle and ray are parallel, if the normal of the triangle and the ray direction are orthoganal, then triangle and ray are parallel

	if (glm::dot(dir, N) > 0) { return false; }// only looks at the front side

	float dot = glm::dot(N, dir);
	if (abs(dot) < FLT_EPSILON) { return false; }

	//D is the distance from the orgin to the plane from Ax + By + Cz + D = 0
	float D = -glm::dot(N, v0);

	//compute t, the distance between camera and the intersection point
	t = -(glm::dot(N, orig) + D) / glm::dot(N, dir);

	//check if triangle is behind the ray
	if (t < 0) { return false; }
	
	//compute P, the intersection point
	glm::vec3 p = t * dir + orig;

	
	if (orig != eyePos && glm::length(p - orig) > glm::length(lightPos - orig)) { return false; }

	//edge 0
	glm::vec3 edge0 = v1 - v0;
	glm::vec3 vp0 = p - v0;
	glm::vec3 C0 = glm::cross(edge0, vp0);//the normal vector of and edge and P;
	if (glm::dot(N, C0) < 0) { return false; }// P is on the right side

	//edge 1
	glm::vec3 edge1 = v2 - v1;
	glm::vec3 vp1 = p - v1;
	glm::vec3 C1 = glm::cross(edge1, vp1);
	if (glm::dot(N, C1) < 0) { return false; } // P is on the right side

	//edge 2
	glm::vec3 edge2 = v0 - v2;
	glm::vec3 vp2 = p - v2;
	glm::vec3 C2 = glm::cross(edge2, vp2);
	if (glm::dot(N, C2) < 0){return false; }// P is on the right side

	//if C and N are going opposite direciton, then P is outside of the triangle
	return true;
}
bool checkIntersection(glm::vec3& orig, glm::vec3 &dir, float & t, int& j) {
	j = 0;
	for (int i = 0;i < shapes_vertices.size() / 3; ++i) {
		glm::vec3 v0 = shapes_vertices[j];
		glm::vec3 v1 = shapes_vertices[j + 1];
		glm::vec3 v2 = shapes_vertices[j + 2];
		if (rayTriangleIntersect(orig, dir, v0, v1, v2, t)) {
			return true;
		}
		j += 3;
	}
	return false;
}
glm::vec4 RayTrace(glm::vec3 &orig, glm::vec3& u, int depth) {
	int shapeIndex;
	float t ;
	glm::vec4 color(0.1f, 0.1f, 0.1f,1.0f);
	//Part I - NoneRecursive computations
	//if no point was intersection return the background color
	if (!checkIntersection(orig, u, t, shapeIndex) ) {
		return color;
	}

	//let z be the first intersection point
	glm::vec3 z(t * u + orig);
	//set n to be the surface normal at the intersection


	
	int shadowIndex=0;
	//shadow feeler from z to the light
	glm::vec3 shadowFeeler(lightPos - z);
	bool shadow = checkIntersection(z, shadowFeeler, t, shadowIndex);

	color = shade(glm::vec4(z, 1.0f), shapes_colors[shapeIndex],glm::normalize( shapes_normal[shapeIndex]), shadow);

	//Part II - Recursive computations
	if (depth == 0) {
		return color;	//Reached maximum trace depth
	}
	
	float prg = 0.3f;


	glm::vec3 n(shapes_normal[shapeIndex]);
	n = glm::normalize(n);
	//Calculate reflection direction and add in reflection color
	if (prg != 0.0f) { //if non zero reflection
		u = glm::normalize(u);
		n = glm::normalize(n);
		glm::vec3 r = u - ((2 * glm::dot(u, n)) * n) ; 
		//glm::vec3 r = glm::reflect(u, n);
		r = glm::normalize(r);
		color = color + prg * RayTrace(z, r, depth - 1); 
	}
	return color;
	
}

void RayTraceMain(float Px,float Py) {
	// let x be the postion of the viewer
	glm::vec3 x(eyePos);  

	// let maxDepth be a positive integer
	int maxDepth = 1;							

	//set u as unit vector in the direction from x to p;
	glm::vec3 P(Px, Py, 0);
	glm::vec3 u(P-x); 
	glm::vec4 color(RayTrace(x, u, maxDepth));
	
	for (int i = 0; i < 6; i++) {
		plane_colors.push_back(color);
	}
}
void CreateShaders( void )
{
	// Renders without any transformations
	PassthroughShader.Create( "./shaders/simple.vert", "./shaders/simple.frag" );

	// Renders using perspective projection
	PerspectiveShader.Create( "./shaders/persp.vert", "./shaders/persp.frag" );

	//
	// Additional shaders would be defined here
	//
}


/*=================================================================================================
	BUFFERS
=================================================================================================*/

void CreateAxisBuffers( void )
{
	glGenVertexArrays( 1, &axis_VAO ); //generate 1 new VAO, its ID is returned in axis_VAO
	glBindVertexArray( axis_VAO ); //bind the VAO so the subsequent commands modify it

	glGenBuffers( 2, &axis_VBO[0] ); //generate 2 buffers for data, their IDs are returned to the axis_VBO array

	// first buffer: vertex coordinates
	glBindBuffer( GL_ARRAY_BUFFER, axis_VBO[0] ); //bind the first buffer using its ID
	glBufferData( GL_ARRAY_BUFFER, sizeof( axis_vertices ), axis_vertices, GL_STATIC_DRAW ); //send coordinate array to the GPU
	glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof( float ), (void*)0 ); //let GPU know this is attribute 0, made up of 4 floats
	glEnableVertexAttribArray( 0 );

	// second buffer: colors
	glBindBuffer( GL_ARRAY_BUFFER, axis_VBO[1] ); //bind the second buffer using its ID
	glBufferData( GL_ARRAY_BUFFER, sizeof( axis_colors ), axis_colors, GL_STATIC_DRAW ); //send color array to the GPU
	glVertexAttribPointer( 1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof( float ), (void*)0 ); //let GPU know this is attribute 1, made up of 4 floats
	glEnableVertexAttribArray( 1 );

	glBindVertexArray( 0 ); //unbind when done

	//NOTE: You will probably not use an array for your own objects, as you will need to be
	//      able to dynamically resize the number of vertices. Remember that the sizeof()
	//      operator will not give an accurate answer on an entire vector. Instead, you will
	//      have to do a calculation such as sizeof(v[0]) * v.size().
}

//
//void CreateMyOwnObject( void ) ...
//
void CreateCylinder(float x, float y, float z, float r, float height) {
	float pi = 3.14159f;
	float n = 6.0f;
	float thetaDif = 2 * pi / n;
	//top and bottom
	for (float theta = 0; theta < 2 * pi; theta += thetaDif) {
		float Ax = r * cos(theta);
		float Ay = 0.0f;
		float Az = r * sin(theta);
		float A_x = r * cos(theta + thetaDif);
		float A_y = 0.0f;
		float A_z = r * sin(theta + thetaDif);
		//bottom
		shapes_vertices.push_back(glm::vec4(Ax + x, Ay + 0.01f + y, Az + z, 1.0f));
		shapes_vertices.push_back(glm::vec4(A_x + x, A_y + 0.01f + y, A_z + z, 1.0f));
		shapes_vertices.push_back(glm::vec4(0.0f + x, 0.01f + y, 0.0f + z, 1.0f));
		//top
		shapes_vertices.push_back(glm::vec4(A_x + x, A_y + y + height, A_z + z, 1.0f));
		shapes_vertices.push_back(glm::vec4(Ax + x, Ay + y + height, Az + z, 1.0f));
		shapes_vertices.push_back(glm::vec4(0.0f + x, height + y, 0.0f + z, 1.0f));
		//wall
		shapes_vertices.push_back(glm::vec4(A_x + x, A_y + y, A_z + z, 1.0f));
		shapes_vertices.push_back(glm::vec4(Ax + x, Ay + y, Az + z, 1.0f));
		shapes_vertices.push_back(glm::vec4(Ax + x, Ay + height + y, Az + z, 1.0f));
		shapes_vertices.push_back(glm::vec4(Ax + x, Ay + height + y, Az + z, 1.0f));
		shapes_vertices.push_back(glm::vec4(A_x + x, A_y + y + height, A_z + z, 1.0f));
		shapes_vertices.push_back(glm::vec4(A_x + x, A_y + y, A_z + z, 1.0f));

		//normal



		glm::vec4 normalA(glm::normalize(glm::vec3(0.0f, -1.0f, 0.0f)), 1.0f);
		glm::vec4 normalB(glm::normalize(glm::vec3(0.0f, -1.0f, 0.0f)), 1.0f);
		glm::vec4 normalC(glm::normalize(glm::vec3(0.0f, -1.0f, 0.0f)), 1.0f);

		shapes_normal.push_back(normalA);
		shapes_normal.push_back(normalB);
		shapes_normal.push_back(normalC);
		normalA = glm::vec4(glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)), 1.0f);
		normalB = glm::vec4(glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)), 1.0f);
		normalC = glm::vec4(glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)), 1.0f);

		shapes_normal.push_back(normalA);
		shapes_normal.push_back(normalB);
		shapes_normal.push_back(normalC);

		glm::vec3 A(Ax, Ay, Az);
		glm::vec3 B(Ax, Ay + height , Az);
		glm::vec3 C(A_x, A_y , A_z);
		glm::vec3 edge1(A - C);
		glm::vec3 edge2(B - C);
		glm::vec4 face_normal(glm::cross(edge1, edge2),1.0f);

		std::cout << glm::length(face_normal) << std::endl;

		for (int i = 0;i < 6; i++) {
			shapes_normal.push_back(face_normal);
		}

		for (int i = 0;i < 12;i++) {
			shapes_colors.push_back(glm::vec4(0.6f, 1.0f, 0.6f, 1.0f));
		}

	}

	glGenVertexArrays(1, &shapes_VAO);
	glBindVertexArray(shapes_VAO);

	glGenBuffers(3, &shapes_VBO[0]);
	//first buffer
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, shapes_vertices.size() * sizeof(glm::vec4), &shapes_vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(0);
	//Second array
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, shapes_colors.size() * sizeof(glm::vec4), &shapes_colors[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(1);
	//thrid array
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[2]);
	glBufferData(GL_ARRAY_BUFFER, shapes_normal.size() * sizeof(glm::vec4), &shapes_normal[0], GL_STATIC_DRAW);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
}
void CreateSphere(float x, float y, float z, float r) {
	float pi = 3.1415f;
	float n = 10.0f;
	float m = n/2;
	float thetaDif = 2 * pi / n;
	float phiDif = pi / m;

	for (float phi = -pi/2;  phi < pi/2; phi += phiDif) {
		for (float theta = 0; theta < 2 * pi; theta += thetaDif) {
			float Ax = r * cos(phi) * cos(theta);
			float Ay = r * cos(phi) * sin(theta);
			float Az = r * sin(phi);

			float A_x = r * cos(phi + phiDif) * cos(theta);
			float A_y = r * cos(phi + phiDif) * sin(theta);
			float A_z = r * sin(phi + phiDif);

			float Bx = r * cos(phi) * cos(theta + thetaDif);
			float By = r * cos(phi) * sin(theta + thetaDif);
			float Bz = r * sin(phi);

			float B_x = r * cos(phi + phiDif) * cos(theta + thetaDif);
			float B_y = r * cos(phi + phiDif) * sin(theta + thetaDif);
			float B_z = r * sin(phi + phiDif);

			glm::vec4 A(Ax + x, Ay + y, Az + z, 1.0f);
			glm::vec4 A_(A_x + x, A_y + y, A_z + z, 1.0f);
			glm::vec4 B(Bx + x, By + y, Bz + z, 1.0f);
			glm::vec4 B_(B_x + x, B_y + y, B_z + z, 1.0f);

			//top and bottom of the sphere only has one triangle instead of two
			if (phi != -pi / 2) {
				shapes_vertices.push_back(A);
				shapes_vertices.push_back(B);
				shapes_vertices.push_back(A_);

				glm::vec3 edge1(B - A);
				glm::vec3 edge2(A_ - A);

				glm::vec4 normal = glm::vec4(glm::cross(edge1, edge2), 1.0f);

				normal = glm::normalize(normal);

				//color
				for (int i = 0; i < 3;i++) {
					shapes_normal.push_back(normal);
					shapes_colors.push_back(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
				}
			}

			if (phi + phiDif < pi /2 ) {
				shapes_vertices.push_back(A_);
				shapes_vertices.push_back(B);
				shapes_vertices.push_back(B_);

				glm::vec3 edge1(B - A_);
				glm::vec3 edge2(B_ - A_);

				glm::vec4 normal = glm::vec4(glm::cross(edge1, edge2), 1.0f);

				normal = glm::normalize(normal);
				//color
				for (int i = 0; i < 3;i++) {
					shapes_normal.push_back(normal);
					shapes_colors.push_back(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
				}
			}
		}
	}
	glGenVertexArrays(1, &shapes_VAO);
	glBindVertexArray(shapes_VAO);

	glGenBuffers(3, &shapes_VBO[0]);
	//first buffer
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, shapes_vertices.size() * sizeof(glm::vec4), &shapes_vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(0);
	//Second array
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, shapes_colors.size() * sizeof(glm::vec4), &shapes_colors[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(1);
	////thrid array
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[2]);
	glBufferData(GL_ARRAY_BUFFER, shapes_normal.size() * sizeof(glm::vec4), &shapes_normal[0], GL_STATIC_DRAW);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
}

void CreateNormLines(void) {
	int j = 0;
	for (int i = 0; i < shapes_vertices.size(); i += 3) {
		glm::vec4 a = shapes_vertices[j];
		glm::vec4 b = shapes_vertices[j + 1];
		glm::vec4 c = shapes_vertices[ j + 2];
		glm::vec3 sum = glm::vec3(a) + glm::vec3(b) + glm::vec3(c);
		glm::vec4 center(sum / 3.0f, 1.0f);
		norm_vertices.push_back(center);

		norm_vertices.push_back(center + (shapes_normal[i] * 0.1f));
		j += 3;
	}

	for (int i = 0; i < norm_vertices.size(); i++) {
		norm_colors.push_back(glm::vec4(0.0f, 1.0f, 1.0f, 1.0f));
	}
	glGenVertexArrays(1, &norm_VAO);
	glBindVertexArray(norm_VAO);

	glGenBuffers(2, &norm_VBO[0]);

	// first buffer: vertex coordinates
	glBindBuffer(GL_ARRAY_BUFFER, norm_VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, norm_vertices.size() * sizeof(glm::vec4), &norm_vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(0);

	// second buffer: colors
	glBindBuffer(GL_ARRAY_BUFFER, norm_VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, norm_colors.size() * sizeof(glm::vec4), &norm_colors[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}
void CreateCuboid(float x, float y, float z, float width, float length, float height) {
	glm::vec4 lowerLeft(-width / 2 + x, y + 0.0f, -length / 2 + z, 1.0f);
	glm::vec4 lowerRight(width / 2 + x, y + 0.0f, -length / 2 + z, 1.0f);
	glm::vec4 topLeft(-width / 2 + x, y + 0.0f, length / 2 + z, 1.0f);
	glm::vec4 topRight(width / 2 + x, y + 0.0f, length / 2 + z, 1.0f);
	glm::vec4 lowerLeftH(-width / 2 + x, y + height, -length / 2 + z, 1.0f);
	glm::vec4 lowerRightH(width / 2 + x, y + height, -length / 2 + z, 1.0f);
	glm::vec4 topLeftH(-width / 2 + x, y + height, length / 2 + z, 1.0f);
	glm::vec4 topRightH(width / 2 + x, y + height, length / 2 + z, 1.0f);

	//floor
	shapes_vertices.push_back(lowerLeft);
	shapes_vertices.push_back(lowerRight);
	shapes_vertices.push_back(topLeft);
	shapes_vertices.push_back(topLeft);
	shapes_vertices.push_back(lowerRight);
	shapes_vertices.push_back(topRight);
	for (int i = 0;i < 6;i++) {
		shapes_normal.push_back(glm::vec4(glm::normalize(glm::vec3(0.0f, -1.0f, 0.0f)), 1.0f));
	}
	//ceiling
	shapes_vertices.push_back(lowerLeftH);
	shapes_vertices.push_back(topLeftH);
	shapes_vertices.push_back(lowerRightH);
	shapes_vertices.push_back(lowerRightH);
	shapes_vertices.push_back(topLeftH);
	shapes_vertices.push_back(topRightH);
	for (int i = 0;i < 6;i++) {
		shapes_normal.push_back(glm::vec4(glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)), 1.0f));
	}
	//wall(1)
	shapes_vertices.push_back(lowerLeft);
	shapes_vertices.push_back(lowerLeftH);
	shapes_vertices.push_back(lowerRight);
	shapes_vertices.push_back(lowerRight);
	shapes_vertices.push_back(lowerLeftH);
	shapes_vertices.push_back(lowerRightH);
	for (int i = 0;i < 6;i++) {
		shapes_normal.push_back(glm::vec4(glm::normalize(glm::vec3(0.0f, 0.0f, -1.0f)), 1.0f));
	}
	//wall(2)
	shapes_vertices.push_back(lowerRight);
	shapes_vertices.push_back(lowerRightH);
	shapes_vertices.push_back(topRight);
	shapes_vertices.push_back(topRight);
	shapes_vertices.push_back(lowerRightH);
	shapes_vertices.push_back(topRightH);
	for (int i = 0;i < 6;i++) {
		shapes_normal.push_back(glm::vec4(glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f)), 1.0f));
	}
	//wall(3)
	shapes_vertices.push_back(topRight);
	shapes_vertices.push_back(topRightH);
	shapes_vertices.push_back(topLeft);
	shapes_vertices.push_back(topLeft);
	shapes_vertices.push_back(topRightH);
	shapes_vertices.push_back(topLeftH);
	for (int i = 0;i < 6;i++) {
		shapes_normal.push_back(glm::vec4(glm::normalize(glm::vec3(0.0f, 0.0f, 1.0f)), 1.0f));
	}
	//wall(4)
	shapes_vertices.push_back(topLeft);
	shapes_vertices.push_back(topLeftH);
	shapes_vertices.push_back(lowerLeft);
	shapes_vertices.push_back(lowerLeft);
	shapes_vertices.push_back(topLeftH);
	shapes_vertices.push_back(lowerLeftH);
	for (int i = 0;i < 6;i++) {
		shapes_normal.push_back(glm::vec4(glm::normalize(glm::vec3(-1.0f, 0.0f, 0.0f)), 1.0f));
	}
	for (int i = 0; i < 36; i++) {
		shapes_colors.push_back(glm::vec4(0.8f, 0.6f, 1.0f, 1.0f));
	}

	glGenVertexArrays(1, &shapes_VAO);
	glBindVertexArray(shapes_VAO);

	glGenBuffers(3, &shapes_VBO[0]);
	//first buffer
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, shapes_vertices.size() * sizeof(glm::vec4), &shapes_vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(0);
	//Second array
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, shapes_colors.size() * sizeof(glm::vec4), &shapes_colors[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(1);
	//thrid array
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[2]);
	glBufferData(GL_ARRAY_BUFFER, shapes_normal.size() * sizeof(glm::vec4), &shapes_normal[0], GL_STATIC_DRAW);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);

}
void CreateFloor(void) {
	//floor
	glm::vec4 bottomLeft(-1.0f, 0.0f, -2.0f, 1.0f);
	glm::vec4 bottomRight(1.0f, 0.0f, -2.0f, 1.0f);
	glm::vec4 topLeft(-1.0f, 0.0f, 0.0f, 1.0f);
	glm::vec4 topRight(1.0f, 0.0f, 0.0f, 1.0f);

	//back wall
	glm::vec4 ceilingLeft(-1.0f, 2.0f, -2.0f, 1.0f);
	glm::vec4 ceilingRight(1.0f, 2.0f, -2.0f, 1.0f);

	//leftwall
	glm::vec4 leftwall(-1.0f, 2.0f, 0.0f, 1.0f);
	//rightwall
	glm::vec4 rightwall(1.0f, 2.0f, 0.0f, 1.0f);


	shapes_vertices.push_back(bottomLeft);
	shapes_vertices.push_back(topLeft);
	shapes_vertices.push_back(bottomRight);
	shapes_vertices.push_back(bottomRight);
	shapes_vertices.push_back(topLeft);
	shapes_vertices.push_back(topRight);

	shapes_vertices.push_back(bottomRight);
	shapes_vertices.push_back(ceilingLeft);
	shapes_vertices.push_back(bottomLeft);
	shapes_vertices.push_back(ceilingLeft);
	shapes_vertices.push_back(bottomRight);
	shapes_vertices.push_back(ceilingRight);

	shapes_vertices.push_back(bottomLeft);
	shapes_vertices.push_back(ceilingLeft);
	shapes_vertices.push_back(topLeft);
	shapes_vertices.push_back(topLeft);
	shapes_vertices.push_back(ceilingLeft);
	shapes_vertices.push_back(leftwall);

	shapes_vertices.push_back(rightwall);
	shapes_vertices.push_back(ceilingRight);
	shapes_vertices.push_back(bottomRight);
	shapes_vertices.push_back(bottomRight);
	shapes_vertices.push_back(topRight);
	shapes_vertices.push_back(rightwall);

	shapes_vertices.push_back(ceilingLeft);
	shapes_vertices.push_back(ceilingRight);
	shapes_vertices.push_back(leftwall);
	shapes_vertices.push_back(leftwall);
	shapes_vertices.push_back(ceilingRight);
	shapes_vertices.push_back(rightwall);




	for (int i = 0; i < 6;i++) {
		shapes_colors.push_back(glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
		shapes_normal.push_back(glm::normalize(glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)));
	}
	for (int i = 0; i < 6;i++) {
		shapes_colors.push_back(glm::vec4(1.0f, 0.5f, 0.5f, 1.0f));
		shapes_normal.push_back(glm::normalize(glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)));
	}

	for (int i = 0; i < 6;i++) {
		shapes_colors.push_back(glm::vec4(0.5f, 1.0f, 0.5f, 1.0f));
		shapes_normal.push_back(glm::normalize(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)));
	}
	for (int i = 0; i < 6;i++) {
		shapes_colors.push_back(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
		shapes_normal.push_back(glm::normalize(glm::vec4(-1.0f, 0.0f, 0.0f, 1.0f)));
	}
	for (int i = 0; i < 6;i++) {
		shapes_colors.push_back(glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
		shapes_normal.push_back(glm::normalize(glm::vec4(0.0f, -1.0f, 0.0f, 1.0f)));
	}

	glGenVertexArrays(1, &shapes_VAO);
	glBindVertexArray(shapes_VAO);

	glGenBuffers(3, &shapes_VBO[0]);
	//first buffer
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, shapes_vertices.size() * sizeof(glm::vec4), &shapes_vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(0);
	//Second array
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, shapes_colors.size() * sizeof(glm::vec4), &shapes_colors[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(1);
	//thrid array
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[2]);
	glBufferData(GL_ARRAY_BUFFER, shapes_normal.size() * sizeof(glm::vec4), &shapes_normal[0], GL_STATIC_DRAW);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);

}
void CreateTriangle() {
	glm::vec4 v0(-0.5f, 0.0f, -1.0f, 1.0f);
	glm::vec4 v1(0.5f, 0.0f, -1.0f, 1.0f);
	glm::vec4 v2(0.0f, 0.5f, -1.0f, 1.0f);
	shapes_vertices.push_back(v0);
	shapes_vertices.push_back(v1);
	shapes_vertices.push_back(v2);

	for (int i = 0; i < 3; i++) {
		shapes_colors.push_back(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
	}

	for (int i = 0; i < 3; i++) {
		shapes_normal.push_back(glm::vec4(glm::normalize(glm::vec3(0.0f, 0.0f, 1.0f)), 1.0f));
	}

	glGenVertexArrays(1, &shapes_VAO);
	glBindVertexArray(shapes_VAO);

	glGenBuffers(3, &shapes_VBO[0]);
	//first buffer
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, shapes_vertices.size() * sizeof(glm::vec4), &shapes_vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(0);
	//Second array
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, shapes_colors.size() * sizeof(glm::vec4), &shapes_colors[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(1);
	//thrid array
	glBindBuffer(GL_ARRAY_BUFFER, shapes_VBO[2]);
	glBufferData(GL_ARRAY_BUFFER, shapes_normal.size() * sizeof(glm::vec4), &shapes_normal[0], GL_STATIC_DRAW);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
}
 
void CreatePlane(float xdif, float ydif, float zdif) {


	float width = 2000;
	float height = 2000;

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			float Px = (2 * ((x + 0.5) / width) - 1) + xdif;
			float Py = (1 - 2 * (y + 0.5) / height) + ydif;
			RayTraceMain(Px,Py);

			float leftBottomX = (2 * ((x) / width) - 1) + xdif;
			float leftBottomY = (1 - 2 * (y) / height) + ydif;

			float rightBottomX = (2 * ((x+1) / width) - 1) + xdif;
			float rightBottomY = (1 - 2 * (y) / height) + ydif;

			float topLeftX = (2 * ((x) / width) - 1) + xdif;
			float topLeftY = (1 - 2 * (y+1) / height) + ydif;

			float topRightX = (2 * ((x+1) / width) - 1) + xdif;
			float topRightY = (1 - 2 * (y+1) / height) + ydif;


			plane_vertices.push_back(glm::vec4(leftBottomX,	leftBottomY,	zdif , 1.0f));
			plane_vertices.push_back(glm::vec4(topLeftX,	topLeftY,		zdif , 1.0f));
			plane_vertices.push_back(glm::vec4(rightBottomX,rightBottomY,	zdif , 1.0f));
			plane_vertices.push_back(glm::vec4(rightBottomX,rightBottomY,	zdif , 1.0f));
			plane_vertices.push_back(glm::vec4(topLeftX,	topLeftY,		zdif , 1.0f));
			plane_vertices.push_back(glm::vec4(topRightX,	topRightY,		zdif , 1.0f));
		}
	}
	glGenVertexArrays(1, &plane_VAO);
	glBindVertexArray(plane_VAO);

	glGenBuffers(2, &plane_VBO[0]);
	//first buffer
	glBindBuffer(GL_ARRAY_BUFFER, plane_VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, plane_vertices.size() * sizeof(glm::vec4), &plane_vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(0);
	//Second array
	glBindBuffer(GL_ARRAY_BUFFER, plane_VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, plane_colors.size() * sizeof(glm::vec4), &plane_colors[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}

/*=================================================================================================
	CALLBACKS
=================================================================================================*/

//-----------------------------------------------------------------------------
// CALLBACK DOCUMENTATION
// https://www.opengl.org/resources/libraries/glut/spec3/node45.html
// http://freeglut.sourceforge.net/docs/api.php#WindowCallback
//-----------------------------------------------------------------------------

void idle_func()
{
	//uncomment below to repeatedly draw new frames
	glutPostRedisplay();
}
void render() {
	glGenVertexArrays(1, &plane_VAO);
	glBindVertexArray(plane_VAO);

	glGenBuffers(2, &plane_VBO[0]);
	//first buffer
	glBindBuffer(GL_ARRAY_BUFFER, plane_VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, plane_vertices.size() * sizeof(glm::vec4), &plane_vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(0);
	//Second array
	glBindBuffer(GL_ARRAY_BUFFER, plane_VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, plane_colors.size() * sizeof(glm::vec4), &plane_colors[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}
void reshape_func( int width, int height )
{
	WindowWidth  = width;
	WindowHeight = height;

	glViewport( 0, 0, width, height );
	glutPostRedisplay();
}
void keyboard_func( unsigned char key, int x, int y )
{
	key_states[ key ] = true;

	switch( key )
	{
		case'w':
		{
			eyePos.z -= 0.1f;
			centerPos.z -= 0.1f;
			for (int i = 0; i < plane_vertices.size();i++) {
				plane_vertices[i].z -= 0.1f;
			}
			render();
			break;
		}
		case's':
		{
			eyePos.z += 0.1f;
			centerPos.z += 0.1f;
			for (int i = 0; i < plane_vertices.size();i++) {
				plane_vertices[i].z += 0.1f;
			}
			render();
			break;
		}
		case'a':
		{
			//eyePos.x -= 0.1f;
			//centerPos.x -= 0.1f;
			//for (int i = 0; i < plane_vertices.size();i++) {
			//	plane_vertices[i].x -= 0.1f;
			//}
			//render();

			eyePos.x -= 0.5f;
			centerPos.x -= 0.5f;
			plane_vertices.clear();
			plane_colors.clear();
			CreatePlane(eyePos.x, eyePos.y, eyePos.z - 1.75f);
			break;
		}
		case'd':
		{
			
			eyePos.x += 0.5f;
			centerPos.x += 0.5f;	
			plane_vertices.clear();
			plane_colors.clear();
			CreatePlane(eyePos.x, eyePos.y, eyePos.z - 1.75f);
			break;
		}
		case'f':
		{
			eyePos.y -= 0.1f;
			centerPos.y -= 0.1f;
			for (int i = 0; i < plane_vertices.size();i++) {
				plane_vertices[i].y -= 0.1f;
			}
			render();
			break;
		}
		case'r':
		{
			eyePos.y += 0.1f;
			centerPos.y += 0.1f;
			for (int i = 0; i < plane_vertices.size();i++) {
				plane_vertices[i].y += 0.1f;
			}
			render();
			break;
		}
		case'1':
		{
			shapes_colors.clear();
			shapes_vertices.clear();
			shapes_normal.clear();
			plane_vertices.clear();
			plane_colors.clear();


			CreateCylinder(1.0f, 1.0f, -2.0f, 0.2f, 0.5f);
			CreateCylinder(-1.0f, 1.0f, -2.0f, 0.2f, 0.5f);
			CreateCuboid(0.0f, 0.0f, -2.0f, 0.5f, 3.0f, 3.0f);

			CreatePlane(eyePos.x, eyePos.y, eyePos.z - 1.75f);
			break;
		}
		case'2':
		{
			shapes_colors.clear();
			shapes_vertices.clear();
			shapes_normal.clear();
			plane_vertices.clear();
			plane_colors.clear();

			CreateCylinder(0.0f, 0.0f, -2.0f, 0.2f, 0.5f);
			CreateFloor();

			CreatePlane(eyePos.x, eyePos.y, eyePos.z - 1.75f);
			break;
		}

		case '0':
		{
			draw_wireframe = !draw_wireframe;
			if( draw_wireframe == true )
				std::cout << "Wireframes on.\n";
			else
				std::cout << "Wireframes off.\n";
			break;
		}

		// Exit on escape key press
		case '\x1B':
		{
			exit( EXIT_SUCCESS );
			break;
		}
	}
}

void key_released( unsigned char key, int x, int y )
{
	key_states[ key ] = false;
}

void key_special_pressed( int key, int x, int y )
{
	key_special_states[ key ] = true;
}

void key_special_released( int key, int x, int y )
{
	key_special_states[ key ] = false;
}

void mouse_func( int button, int state, int x, int y )
{
	// Key 0: left button
	// Key 1: middle button
	// Key 2: right button
	// Key 3: scroll up
	// Key 4: scroll down

	if( x < 0 || x > WindowWidth || y < 0 || y > WindowHeight )
		return;

	float px, py;
	window_to_scene( x, y, px, py );

	if( button == 3 )
	{
		perspZoom += 0.03f;
	}
	else if( button == 4 )
	{
		if( perspZoom - 0.03f > 0.0f )
			perspZoom -= 0.03f;
	}

	mouse_states[ button ] = ( state == GLUT_DOWN );

	LastMousePosX = x;
	LastMousePosY = y;
}

void passive_motion_func( int x, int y )
{
	if( x < 0 || x > WindowWidth || y < 0 || y > WindowHeight )
		return;

	float px, py;
	window_to_scene( x, y, px, py );

	LastMousePosX = x;
	LastMousePosY = y;
}

void active_motion_func( int x, int y )
{
	if( x < 0 || x > WindowWidth || y < 0 || y > WindowHeight )
		return;

	float px, py;
	window_to_scene( x, y, px, py );

	if( mouse_states[0] == true )
	{
		perspRotationY += ( x - LastMousePosX ) * perspSensitivity;
		perspRotationX += ( y - LastMousePosY ) * perspSensitivity;
	}
	LastMousePosX = x;
	LastMousePosY = y;
}

/*=================================================================================================
	RENDERING
=================================================================================================*/

void display_func( void )
{
	// Clear the contents of the back buffer
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	// Update transformation matrices
	CreateTransformationMatrices();

	// Choose which shader to user, and send the transformation matrix information to it
	PerspectiveShader.Use();
	PerspectiveShader.SetUniform( "projectionMatrix", glm::value_ptr( PerspProjectionMatrix ), 4, GL_FALSE, 1 );
	PerspectiveShader.SetUniform( "viewMatrix", glm::value_ptr( PerspViewMatrix ), 4, GL_FALSE, 1 );
	PerspectiveShader.SetUniform( "modelMatrix", glm::value_ptr( PerspModelMatrix ), 4, GL_FALSE, 1 );

	// Drawing in wireframe?
	if( draw_wireframe == true )
		glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	else
		glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

	// Bind the axis Vertex Array Object created earlier, and draw it
	glBindVertexArray( axis_VAO );
	glDrawArrays( GL_LINES, 0, 6 ); // 6 = number of vertices in the object

	//
	// Bind and draw your object here
	//

	glBindVertexArray(shapes_VAO);
	glDrawArrays(GL_TRIANGLES, 0, shapes_vertices.size());

	//plane
	glBindVertexArray(plane_VAO);
	glDrawArrays(GL_TRIANGLES, 0, plane_vertices.size());

	//norm
	glBindVertexArray(norm_VAO);
	glDrawArrays(GL_LINES, 0, norm_vertices.size());
	glBindVertexArray(0);


	// Swap the front and back buffers
	glutSwapBuffers();
}


/*=================================================================================================
	INIT
=================================================================================================*/

void init( void )
{
	// Print some info
	std::cout << "Vendor:         " << glGetString( GL_VENDOR   ) << "\n";
	std::cout << "Renderer:       " << glGetString( GL_RENDERER ) << "\n";
	std::cout << "OpenGL Version: " << glGetString( GL_VERSION  ) << "\n";
	std::cout << "GLSL Version:   " << glGetString( GL_SHADING_LANGUAGE_VERSION ) << "\n\n";

	// Set OpenGL settings
	glClearColor( 0.0f, 0.0f, 0.0f, 0.0f ); // background color
	glEnable( GL_DEPTH_TEST ); // enable depth test
	glEnable( GL_CULL_FACE ); // enable back-face culling

	// Create shaders
	CreateShaders();

	// Create axis buffers
	CreateAxisBuffers();

	//
	// Consider calling a function to create your object here
	//
	//case 1
	/*CreateCylinder(1.0f, 1.0f, -2.0f, 0.2f, 0.5f);
	CreateCylinder(-1.0f, 1.0f, -2.0f, 0.2f, 0.5f);
	CreateCuboid(0.0f, 0.0f, -2.0f, 0.5f, 3.0f, 3.0f);

	CreatePlane(eyePos.x, eyePos.y, eyePos.z - 1.75f);*/
	
	//case 2 
	// 
	
	/*CreateCylinder(0.5f, 0.0f, -1.0f, 0.2f, 0.5f);
	CreateCylinder(-0.5f, 0.0f, -1.0, 0.2f, 0.5f);*/
	
	CreateCuboid(0.5f, 0.0f, -1.0f, 0.2f, 0.2f, 0.2f);
	CreateCuboid(-0.5f, 0.0f, -1.0f, 0.2f, 0.2f, 0.2f);

	CreateFloor();

	CreatePlane(eyePos.x, eyePos.y, eyePos.z - 1.75f);

	

	//CreateSphere(lightPos.x, lightPos.y, lightPos.z, 0.1f);
	CreateNormLines();
	std::cout << "Finished initializing...\n\n";
}

/*=================================================================================================
	MAIN
=================================================================================================*/

int main( int argc, char** argv )
{
	// Create and initialize the OpenGL context
	glutInit( &argc, argv );

	glutInitWindowPosition( 100, 100 );
	glutInitWindowSize( InitWindowWidth, InitWindowHeight );
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH );

	glutCreateWindow( "CSE-170 Computer Graphics" );

	// Initialize GLEW
	GLenum ret = glewInit();
	if( ret != GLEW_OK ) {
		std::cerr << "GLEW initialization error." << std::endl;
		glewGetErrorString( ret );
		return -1;
	}

	// Register callback functions
	glutDisplayFunc( display_func );
	glutIdleFunc( idle_func );
	glutReshapeFunc( reshape_func );
	glutKeyboardFunc( keyboard_func );
	glutKeyboardUpFunc( key_released );
	glutSpecialFunc( key_special_pressed );
	glutSpecialUpFunc( key_special_released );
	glutMouseFunc( mouse_func );
	glutMotionFunc( active_motion_func );
	glutPassiveMotionFunc( passive_motion_func );


	// Do program initialization
	init();

	// Enter the main loop
	glutMainLoop();

	return EXIT_SUCCESS;
}
