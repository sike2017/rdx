#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include "mesh.h"

typedef std::pair<std::string, material*> mtl_pair;
typedef std::map<std::string, material*> mtl_map;

class FileParser
{
public:
	virtual bool parse(const std::string& path, Mesh* _mesh) { return false; }
	virtual bool parse(const std::string& path, mtl_map** _list) { return false; }

protected:
	inline bool openFile(const std::string& filename) {
		moduleFile = std::ifstream(filename, std::ios::in);
		if (!moduleFile.is_open()) {
			//rlog << "open file error\n";
			return false;
		}
		return true;
	}
	inline void closeFile() {
		moduleFile.close();
	}

	template<typename T>
	inline T convertStringToNumber(const std::string& s) {
		std::istringstream istream(s);
		T r;
		istream >> r;
		return r;
	}

	std::ifstream moduleFile;
};

class MtlParser : public FileParser
{
public:
	MtlParser() {}
	~MtlParser() {}

	virtual bool parse(const std::string& path, mtl_map** _list) {
		if (!openFile(path)) {
			return false;
		}
		
		mtl_map* list = new mtl_map;
		*_list = list;
		std::string strLine;
		while (std::getline(moduleFile, strLine)) {
			line_parse(strLine, list);
		}
	}

private:
	void divid_space(const std::string& line, base_list<std::string>* list) {
		std::string temp;
		int index = 0;
		for (auto& letter : line) {
			index++;
			if (letter != ' ' && index != line.length()) {
				temp.push_back(letter);
			}
			else {
				if (index == line.length()) {
					temp.push_back(letter);
				}
				if (!temp.empty()) {
					list->push_back(temp);
					temp.clear();
				}
				continue;
			}
		}
	}

	bool line_parse(const std::string& line, mtl_map* pout) {
		int index = 0;
		base_list<std::string> list;
		divid_space(line, &list);
		size_t size = list.size();
		float rdx_light;
		std::string strInList;
		while (index < size) {
			strInList = list[index];
			if (strInList == "#") {
				if (index > size - 4) {
					break;
				}
				std::string temp = list[index + 1];
				if (temp != "__rdx__") {
					break;
				}
				temp = list[index + 2];
				if (temp == "light") {
					rdx_light = convertStringToNumber<float>(list[index + 3]);
					pout->find(matr_name)->second = new diffuse_light(new solid_texture(Color(1.0, 1.0, 1.0) * rdx_light));
					return true;
				}
			}
			else if (strInList == "newmtl") {
				createMode = true;
				material* current_material = new lambertian(new solid_texture(Color(0.7, 0.7, 0.7)));
				pout->insert(mtl_pair(list[index + 1], current_material));
				matr_name = list[index + 1];
			}
			//else if (strInList == "Ns") {
			//	matr->ns = convertStringToNumber<float>(list[index + 1]);
			//	return true;
			//}
			//else if (strInList == "Ka") {
			//	matr->ka[0] = convertStringToNumber<float>(list[index + 1]);
			//	matr->ka[1] = convertStringToNumber<float>(list[index + 2]);
			//	matr->ka[2] = convertStringToNumber<float>(list[index + 3]);
			//	return true;
			//}
			//else if (strInList == "Ks") {
			//	matr->ks[0] = convertStringToNumber<float>(list[index + 1]);
			//	matr->ks[1] = convertStringToNumber<float>(list[index + 2]);
			//	matr->ks[2] = convertStringToNumber<float>(list[index + 3]);
			//	return true;
			//}
			//else if (strInList == "Ke") {
			//	matr->ke[0] = convertStringToNumber<float>(list[index + 1]);
			//	matr->ke[1] = convertStringToNumber<float>(list[index + 2]);
			//	matr->ke[2] = convertStringToNumber<float>(list[index + 3]);
			//	return true;
			//}
			//else if (strInList == "Ni") {
			//	matr->ni = convertStringToNumber<float>(list[index + 1]);
			//	return true;
			//}
			//else if (strInList == "d") {
			//	matr->d = convertStringToNumber<float>(list[index + 1]);
			//	return true;
			//}
			//else if (strInList == "illum") {
			//	matr->illum = convertStringToNumber<float>(list[index + 1]);
			//	return true;
			//}
			index++;
		}
		return false;
	}

	bool createMode = false;
	std::string matr_name;
};

class ObjParser : public FileParser
{
public:
	ObjParser() {
		table.insert(std::pair<std::string, ParserInstruction>("v", ParserInstruction::v));
		table.insert(std::pair<std::string, ParserInstruction>("f", ParserInstruction::f));
		table.insert(std::pair<std::string, ParserInstruction>("vt", ParserInstruction::vt));
		table.insert(std::pair<std::string, ParserInstruction>("vn", ParserInstruction::vn));
		table.insert(std::pair<std::string, ParserInstruction>("mtllib", ParserInstruction::mtllib));
		table.insert(std::pair<std::string, ParserInstruction>("usemtl", ParserInstruction::usemtl));
		table.insert(std::pair<std::string, ParserInstruction>("#", ParserInstruction::comment));
	}
	~ObjParser() {}

	enum class ParserInstruction {
		v,
		f,
		vt,
		vn,
		mtllib,
		usemtl,
		comment
	};

	virtual bool parse(const std::string& _filename, Mesh* mesh) {
		objPath = _filename;
		texture = "";
		mtlFound = false;
		textureFound = false;
		if (!openFile(_filename)) {
			return false;
		}
		std::string lineStr;
		rdxr_texture* solidTexture = new solid_texture(Color(0.7, 0.7, 0.7));
		current_matr = new lambertian(solidTexture);
		base_list<Vertex>& vArray = mesh->vArray;
		base_list<Point2f>& vtArray = mesh->vtArray;
		base_list<Vector3f>& vnArray = mesh->vnArray;
		base_list<float> v_vector; // index of vertex element v vt vn
		while (std::getline(moduleFile, lineStr)) {
			std::pair<ParserInstruction, base_list<float>> lineResult;
			if (lineParse(lineStr, &lineResult, mesh)) {
				if (lineResult.first == ParserInstruction::v) {
					vArray.push_back(Vertex(Point3f(lineResult.second[0], lineResult.second[1], lineResult.second[2], 1.0f)));
				}
				else if (lineResult.first == ParserInstruction::f) {
					int vertex_n = 3;
					Vertex *vertex0, *vertex1, *vertex2;
					switch (lineResult.second.size()) {
					case 3:
						// f v v v
						vertex_n = 3;
						for (int index = 0; index < vertex_n; index++) {
							v_vector.push_back(lineResult.second[index]);
							v_vector.push_back(0);
							v_vector.push_back(0);
						}
						vtArray.push_back(Point2f(0, 0));
						vertex0 = &vArray[lineResult.second[0]-1];
						vertex1 = &vArray[lineResult.second[1]-1];
						vertex2 = &vArray[lineResult.second[2]-1];
						vnArray.push_back(Vector3f(computeNormal(vertex0->p, vertex1->p, vertex2->p)));
						break;
					case 4:
						// f v v v v
						vertex_n = 4;
						for (int index = 0; index < vertex_n; index++) {
							v_vector.push_back(lineResult.second[index]);
							v_vector.push_back(0);
							v_vector.push_back(0);
						}
						vtArray.push_back(Point2f(0, 0));
						vertex0 = &vArray[lineResult.second[0] - 1];
						vertex1 = &vArray[lineResult.second[1] - 1];
						vertex2 = &vArray[lineResult.second[2] - 1];
						vnArray.push_back(Vector3f(computeNormal(vertex0->p, vertex1->p, vertex2->p)));
						break;
					default:
						// f v/vt/vn v/vt/vn v/vt/vn v/vt/vn ....
						vertex_n = lineResult.second.size() / 3;
						v_vector = lineResult.second;
					}
					int triangle_n = vertex_n - 2;
					int v0;
					int v1;
					int v2;
					int v[3];
					int vt[3];
					int vn[3];
					for (int index = 0; index < triangle_n; index++) {
						v0 = 0;
						v1 = 3 * (1 + index);
						v2 = 3 * (2 + index);
						v[0] = v_vector[v0] - 1;
						v[1] = v_vector[v1] - 1;
						v[2] = v_vector[v2] - 1;
						vt[0] = v_vector[v0 + 1] - 1;
						vt[1] = v_vector[v1 + 1] - 1;
						vt[2] = v_vector[v2 + 1] - 1;
						vn[0] = v_vector[v0 + 2] - 1;
						vn[1] = v_vector[v1 + 2] - 1;
						vn[2] = v_vector[v2 + 2] - 1;
						mesh->add(
							Triangle(Triple<Vertex*>(&vArray[v[0]], &vArray[v[1]], &vArray[v[2]]), Triple<Point2f*>(&vtArray[vt[0]], &vtArray[vt[1]], &vtArray[vt[2]]), Triple<Vector3f*>(&vnArray[vn[0]], &vnArray[vn[1]], &vnArray[vn[2]]), current_matr));
					}
				}
				else if (lineResult.first == ParserInstruction::vn) {
					vnArray.push_back(Vector3f(lineResult.second[0], lineResult.second[1], lineResult.second[2], 0.0f));
				}
				else if (lineResult.first == ParserInstruction::vt) {
					vtArray.push_back(Point2f(lineResult.second[0], lineResult.second[1]));
				}
				else if (lineResult.first == ParserInstruction::comment) {
					if (textureFound) {
						rdxr_texture* imageTexture = new image_texture(texture);
						current_matr = new lambertian(imageTexture);
					}
				}
			}
		}
		closeFile();
		return true;
	}

private:
	inline bool isFloatString(const std::string& str) {
		for (const char& le : str) {
			if (le != '-' && le != '.' && le != '0' && le != '1' && le != '2' && le != '3' && le != '4'
				&& le != '5' && le != '6' && le != '7' && le != '8' && le != '9' && le != '+' && le != 'e' && le != ' ') {
				return false;
			}
		}
		return true;
	}

	inline bool lineParse(const std::string& str, std::pair<ParserInstruction, base_list<float>>* out, Mesh* mesh) {
		base_list<std::string> e;
		std::string temp;
		int index = 0;

		for (const char& l : str) {
			index++;
			if (l != ' ' && index != str.length()) {
				temp.push_back(l);
			}
			else {
				if (index == str.length()) {
					temp.push_back(l);
				}
				if (!temp.empty()) {
					e.push_back(temp);
					temp.clear();
				}
				continue;
			}
		}
		if (e.empty()) {
			return false;
		}

		std::map<std::string, ParserInstruction>::iterator itor = table.find(e[0]);
		if (itor == table.end()) {
			return false;
		}
		bool first;
		int floatCount;
		std::string fname;
		size_t n;
		switch (itor->second) {
		case ParserInstruction::v:
			vLineParse(e, itor, out);
			break;
		case ParserInstruction::f:
-			fLineParse(e, itor, out);
			break;
		case ParserInstruction::vt:
			first = true;
			floatCount = 0;
			for (const std::string& l : e) {
				if (first) {
					first = false;
					continue;
				}
				if (floatCount == 2) {
					break;
				}
				if (!isFloatString(l)) {
					return false;
				}
				out->second.push_back(convertStringToNumber<float>(l));
				floatCount++;
			}
			out->first = itor->second;
			break;
		case ParserInstruction::vn:
			first = true;
 			floatCount = 0;
			for (const std::string& l : e) {
				if (first) {
					first = false;
					continue;
				}
				if (floatCount == 3) {
					break;
				}
				if (!isFloatString(l)) {
					return false;
				}
				out->second.push_back(convertStringToNumber<float>(l));
				floatCount++;
			}
			out->first = itor->second;
			break;
		case ParserInstruction::mtllib:
			n = objPath.find_last_of('/');
			if (n != std::string::npos) {
				fname = objPath.substr(0, n + 1) + e[1];
			}
			else {
				fname = e[1];
			}
			mtlFound = parser.parse(fname, &mat_map);
			out->first = itor->second;
			break;
		case ParserInstruction::usemtl:
			if (mtlFound == false) {
				// no .mtl file found
				out->first = itor->second;
				break;
			}
			current_matr = mat_map->find(e[1])->second;
			out->first = itor->second;
			break;
		case ParserInstruction::comment:
			if (e.size() >= 4 && e[1] == "__rdx__") {
				if (e[2] == "texture") {
					n = objPath.find_last_of('/');
					if (n != std::string::npos) {
						texture = objPath.substr(0, n + 1) + e[3].substr(1, e[3].size() - 2);
						textureFound = true;
					}
				}
			}
			out->first = itor->second;
		}

		return true;
	}

	inline bool vLineParse(const base_list<std::string>& e, const std::map<std::string, ParserInstruction>::iterator& itor, std::pair<ParserInstruction, base_list<float>>* out) {
		bool first = true;
		int floatCount = 0;
		for (const std::string& l : e) {
			if (first) {
				first = false;
				continue;
			}
			if (floatCount == 3) {
				break;
			}
			if (!isFloatString(l)) {
				return false;
			}
			out->second.push_back(convertStringToNumber<float>(l));
			floatCount++;
		}
		out->first = itor->second;
		return true;
	}

	inline bool fLineParse(const base_list<std::string>& e, const std::map<std::string, ParserInstruction>::iterator& itor, std::pair<ParserInstruction, base_list<float>>* out) {
		bool first = true;
		for (const std::string& l : e) {
			if (first) {
				first = false;
				continue;
			}
			std::string vString;
			int index = 0;
			for (const char& tl : l) {
				index++;
				if (tl != '/') {
					if (tl != '0' && tl != '1' && tl != '2' && tl != '3' && tl != '4'
						&& tl != '5' && tl != '6' && tl != '7' && tl != '8' && tl != '9' && tl != ' ') {
						return false;
					}
					else {
						vString += tl;
						if (index == l.size()) {
							out->second.push_back(convertStringToNumber<int>(vString));
							out->first = itor->second;
							vString.clear();
						}
					}
				}
				else {
					out->second.push_back(convertStringToNumber<int>(vString));
					vString.clear();
				}
			}
		}
		return true;
	}

	std::map<std::string, ParserInstruction> table;
	MtlParser parser;
	material* current_matr;
	mtl_map* mat_map;
	std::string objPath;
	std::string texture;
	bool mtlFound;
	bool textureFound;
};

