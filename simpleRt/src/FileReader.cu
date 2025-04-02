#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "camera.cuh"
#include "material.cuh"
#include "sphere.cuh"
#include "triangle.cuh"
#include <stdio.h>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <tinyobjloader/tiny_obj_loader.h>
#include <string>
#include <neural-graphics-primitives/common.h>
// #include <fmt/core.h>

#include "FileReader.cuh"

using namespace rapidjson;
using namespace std;

// http://tsitsul.in/pdf/colors/normal_12.pdf
static int COLOR_SCHEME[14][3] = {{235, 172, 35},
                     {184, 0, 88},
					 {0, 140, 249},
					 {0, 110, 0},
					 {0, 187, 173},
					 {209, 99, 230},
					 {209,99,230},
					 {178, 69, 2},
					 {255, 146, 135},
					 {89, 84, 214},
					 {0, 198, 248},
					 {135, 133, 0},
					 {0, 167, 108},
					 {189, 189, 189}};



void FileReader::read_obj_file(int index, char *dir, vector<hittable*> &vec_obj_list, material *mat_ptr, float posx, float posy, float posz, bool with_normals) {
	std::string filename(dir);
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());

	if (!warn.empty()) {
		tlog::warning() << "Obj: " << warn;
	}

	if (!err.empty()) {
		printf("error!!!!\n");
		// throw std::runtime_error{fmt::format("Error loading obj: {}", err)};
	}

	bool is_transformed = true;


	// AGAO - Scale and offset mesh in order to match that transform applied to the
	// raw training data by Instant-NGP.  (Hardcode default values for now)
	const float scale = 1 / 3.0;
	const float offset = 0.5;

    const float theta_x = -M_PI / 2.0f; // -90 degrees around x-axis
    const float theta_z = -M_PI / 2.0f; // -90 degrees around z-axis
    const Eigen::AngleAxisf rot_x(theta_x, Eigen::Vector3f::UnitX());
    const Eigen::AngleAxisf rot_z(theta_z, Eigen::Vector3f::UnitZ());
    const Eigen::Matrix3f rot_mat = (rot_x * rot_z).toRotationMatrix();

	// printf("11 vec_obj_list size %d\n", vec_obj_list.size());
	// Loop over shapes
	for (size_t s = 0; s < shapes.size(); s++) {
		// Loop over faces
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

			if (shapes[s].mesh.num_face_vertices[f] != 3) {
				tlog::warning() << "Non-triangle face found in " << filename;
				index_offset += fv;
				continue;
			}

			// Loop over vertices in the face.
			vector<vec3> points;
			vector<vec3> normals;
			for (size_t v = 0; v < 3; v++) {

				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				const tinyobj::real_t vx = attrib.vertices[3*idx.vertex_index+0];
				const tinyobj::real_t vy = attrib.vertices[3*idx.vertex_index+1];
				const tinyobj::real_t vz = attrib.vertices[3*idx.vertex_index+2];

				if (is_transformed) {
					const Eigen::Vector3f point(vx * scale + offset + posx, vy * scale + offset + posy, vz * scale + offset + posz);
					const Eigen::Vector3f rotated_point = rot_mat * point;

					points.push_back(vec3(rotated_point[0], rotated_point[1], rotated_point[2]));
				}
				else {
					points.push_back(vec3(vx + posx, vy + posy, vz + posz));
				}
				if (with_normals) {
				    if (idx.normal_index >= 0) {
				    	const tinyobj::real_t nx = attrib.normals[3*idx.normal_index+0];
				    	const tinyobj::real_t ny = attrib.normals[3*idx.normal_index+1];
				    	const tinyobj::real_t nz = attrib.normals[3*idx.normal_index+2];
				    	normals.push_back(vec3(nx, ny, nz));
				    }
				}
			}
			// printf("\n");

			index_offset += fv;

			if (!with_normals) {
				vec_obj_list.push_back(
					new triangle(points[0], points[1], points[2],
					COLOR_SCHEME[index][0], COLOR_SCHEME[index][1], COLOR_SCHEME[index][2], 
					mat_ptr));
			} else {
				vec_obj_list.push_back(
					new triangle(points[0], points[1], points[2],
					COLOR_SCHEME[index][0], COLOR_SCHEME[index][1], COLOR_SCHEME[index][2], 
					normals[0], normals[1], normals[2],
					mat_ptr));
			}

			// vec_obj_list.push_back(
			// 	new triangle(points[0], points[2], points[1],
			// 	mat_ptr));

		}
	}
}


// void FileReader::read_obj_file(char *dir, vector<hittable*> &vec_obj_list, material *mat_ptr) {
// 	FILE *f;
// 	std::cout << "!!!!~~~~=========== 1\n";
// 	std::cout << dir << "\n";
// 	fopen_s(&f, dir, "r");
// 	if (f == NULL) {
// 		std::cout << "!!!!=========== NULL\n";
// 		exit(9);
// 	}
// 	std::cout << "!!!!=========== 2\n";
// 	vector<vec3> points, p_normals;
// 	//vector<hittable*> vec_objs;
// 	char line[100];
// 	//int obj_id = 0;
// 	if (f != NULL) {
// 		while (fgets(line, 100, f) != NULL) {
// 			int p = 0;
// 			while ((line[p] >= 'a'&&line[p] <= 'z') || (line[p] >= 'A'&&line[p] <= 'Z')) p++;
// 			/*if (line[0] == 'o') {
// 			obj_id++;
// 			}*/
// 			if (line[0] == 'v'&&line[1] != 'n') {
// 				double x, y, z;
// 				sscanf(line + p, "%lf%lf%lf", &x, &y, &z);
// 				vec3 temp = vec3(x, y, z);
// 				//cout<<x*500<<' '<<y*500<<' '<<z*500<<endl;
// 				points.push_back(temp);
// 			}
// 			else if (line[0] == 'v'&&line[1] == 'n') {
// 				double x, y, z;
// 				//has_normal = true;
// 				sscanf(line + p, "%lf%lf%lf", &x, &y, &z);
// 				vec3 temp = vec3(x, y, z);
// 				//cout<<x*500<<' '<<y*500<<' '<<z*500<<endl;
// 				p_normals.push_back(temp);
// 			}
// 			else if (line[0] == 'f') {
// 				int x, y, z;
// 				sscanf(line + p, "%d%d%d", &x, &y, &z);
// 				vec_obj_list.push_back(
// 					new triangle(points[x - 1], points[y - 1], points[z - 1],
// 					mat_ptr));
// 			}
// 		}

// 		fclose(f);
// 	}
// }

bool FileReader::readfile_to_render(
	vector<hittable*>& vec_obj_list,
	vector<hittable*>& vec_lightsrc_list,
	vector<hittable*>& vec_geom_list,
	const char *path,          // 
	int &nx, int &ny, int &ns, // 
	camera *&c, // 
	int &n_bounce
	)
{
	ifstream inputfile(path);
	IStreamWrapper _TMP_ISW(inputfile);
	Document json_tree;
	json_tree.ParseStream(_TMP_ISW);
	cerr << (nx = json_tree["nx"].GetInt()) << endl;
	cerr << (ny = json_tree["ny"].GetInt()) << endl;
	cerr << (ns = json_tree["ns"].GetInt()) << endl;
	cerr << (n_bounce = json_tree["n_bounce"].GetInt()) << endl;

	if (c == NULL)
		c = new camera(
			vec3(
				json_tree["camera"]["lookfrom"][0].GetDouble(),
				json_tree["camera"]["lookfrom"][1].GetDouble(),
				json_tree["camera"]["lookfrom"][2].GetDouble()),
			vec3(
				json_tree["camera"]["lookat"][0].GetDouble(),
				json_tree["camera"]["lookat"][1].GetDouble(),
				json_tree["camera"]["lookat"][2].GetDouble()),
			vec3(
				json_tree["camera"]["vup"][0].GetDouble(),
				json_tree["camera"]["vup"][1].GetDouble(),
				json_tree["camera"]["vup"][2].GetDouble()),
			json_tree["camera"]["vfov"].GetDouble(),
			json_tree["camera"]["aspect"].GetDouble(),
			json_tree["camera"]["aperture"].GetDouble(),
			json_tree["camera"]["focus_dist"].GetDouble());

	int m_list_size = json_tree["materials"].Size();
	material **mat_list = new material *[m_list_size];

	// 0 render; 1 light source; 2 shadow
	int *mat_type = new int[m_list_size];

	for (int i = 0; i < m_list_size; i++)
	{
		mat_type[i] = 0;
		if (strcmp(json_tree["materials"][i]["type"].GetString(), "lambertian") == 0)
		{
			mat_list[i] = new lambertian(
				vec3(
					json_tree["materials"][i]["albedo"][0].GetDouble(),
					json_tree["materials"][i]["albedo"][1].GetDouble(),
					json_tree["materials"][i]["albedo"][2].GetDouble()));
		}
		else if ((strcmp(json_tree["materials"][i]["type"].GetString(), "metal") == 0))
		{
			mat_list[i] = new metal(
				vec3(
					json_tree["materials"][i]["albedo"][0].GetDouble(),
					json_tree["materials"][i]["albedo"][1].GetDouble(),
					json_tree["materials"][i]["albedo"][2].GetDouble()),
				json_tree["materials"][i]["fuzz"].GetDouble());
		}
		else if (((strcmp(json_tree["materials"][i]["type"].GetString(), "dieletric") == 0)))
		{
			mat_list[i] = new dielectric(
				vec3(
					json_tree["materials"][i]["albedo"][0].GetDouble(),
					json_tree["materials"][i]["albedo"][1].GetDouble(),
					json_tree["materials"][i]["albedo"][2].GetDouble()),
				json_tree["materials"][i]["ref_idx"].GetDouble());
		}
		else if (((strcmp(json_tree["materials"][i]["type"].GetString(), "lightsource") == 0)))
		{
			mat_list[i] = new lightsource(
				vec3(
					json_tree["materials"][i]["albedo"][0].GetDouble(),
					json_tree["materials"][i]["albedo"][1].GetDouble(),
					json_tree["materials"][i]["albedo"][2].GetDouble()));

			mat_type[i] = 1;
		}
		else if (((strcmp(json_tree["materials"][i]["type"].GetString(), "shadow") == 0)))
		{
			mat_list[i] = new lambertian(vec3(0, 0., 0.));
			mat_type[i] = 2;
		}
	}
	int sphere_cnt = json_tree["spheres"].Size(); 
	int obj_cnt = json_tree["objfile"].Size();
	//cout << (o_list_size = sphere_cnt + obj_cnt) << endl;
	//obj_list = new hittable *[sphere_cnt + obj_cnt];
	
	for (int i = 0; i < sphere_cnt; i++)
	{
		if (mat_type[json_tree["spheres"][i]["material"].GetInt() - 1] == 1) {
			vec_lightsrc_list.push_back(
				new sphere(
					vec3(
						json_tree["spheres"][i]["center"][0].GetDouble(),
						json_tree["spheres"][i]["center"][1].GetDouble(),
						json_tree["spheres"][i]["center"][2].GetDouble()),
					json_tree["spheres"][i]["radius"].GetDouble(),
					mat_list[json_tree["spheres"][i]["material"].GetInt() - 1]
				)
			);
		}
		else if (mat_type[json_tree["spheres"][i]["material"].GetInt() - 1] == 2) {
			vec_geom_list.push_back(
				new sphere(
					vec3(
						json_tree["spheres"][i]["center"][0].GetDouble(),
						json_tree["spheres"][i]["center"][1].GetDouble(),
						json_tree["spheres"][i]["center"][2].GetDouble()),
					json_tree["spheres"][i]["radius"].GetDouble(),
					mat_list[json_tree["spheres"][i]["material"].GetInt() - 1]
				)
			);
		}
		else {
			vec_obj_list.push_back(
				new sphere(
					vec3(
						json_tree["spheres"][i]["center"][0].GetDouble(),
						json_tree["spheres"][i]["center"][1].GetDouble(),
						json_tree["spheres"][i]["center"][2].GetDouble()),
					json_tree["spheres"][i]["radius"].GetDouble(),
					mat_list[json_tree["spheres"][i]["material"].GetInt() - 1]
				)
			);
		}


	}

	//o_list_size = sphere_cnt;
	for (int i = 0; i < obj_cnt; i++) {
		float posx = 0.f;
		float posy = 0.f;
		float posz = 0.f;
		if (json_tree["objfile"][i].HasMember("position")) {
			posx = json_tree["objfile"][i]["position"][0].GetFloat();
			posy = json_tree["objfile"][i]["position"][1].GetFloat();
			posz = json_tree["objfile"][i]["position"][2].GetFloat();
			std::cout << "Position: " << posx << ", " << posy << ", " << posz << std::endl;
		}

		if (mat_type[json_tree["objfile"][i]["material"].GetInt() - 1] == 1) {
			read_obj_file(
				i,
				(char*)(json_tree["objfile"][i]["dir"].GetString()),
				vec_lightsrc_list,
				mat_list[json_tree["objfile"][i]["material"].GetInt() - 1],
				posx,
				posy,
				posz,
				true
			);
		}
		else if (mat_type[json_tree["objfile"][i]["material"].GetInt() - 1] == 2) {
			read_obj_file(
				i,
				(char*)(json_tree["objfile"][i]["dir"].GetString()),
				vec_geom_list,
				mat_list[json_tree["objfile"][i]["material"].GetInt() - 1],
				posx,
				posy,
				posz,
				true
			);
		} else {
			read_obj_file(
				i,
				(char*)(json_tree["objfile"][i]["dir"].GetString()),
				vec_obj_list,
				mat_list[json_tree["objfile"][i]["material"].GetInt() - 1],
				posx,
				posy,
				posz,
				true
			);
		}

	}
	free(mat_list);




	return 1;
}