#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include "core/mesh.h"

class FileParser
{
public:
	virtual bool parse(const std::string& path, Mesh** _mesh) = 0;

protected:
	inline bool openFile(const std::string& filename) {
		moduleFile = std::ifstream(filename, std::ios::in);
		if (!moduleFile.is_open()) {
			rlog << "open module file error\n";
			return false;
		}
		return true;
	}
	inline void closeFile() {
		moduleFile.close();
	}

	std::ifstream moduleFile;
};

class ObjParser : public FileParser
{
public:
	ObjParser() : FileParser() {
		table.insert(std::pair<std::string, ParserInstruction>("v", ParserInstruction::v));
		table.insert(std::pair<std::string, ParserInstruction>("f", ParserInstruction::f));
		table.insert(std::pair<std::string, ParserInstruction>("vt", ParserInstruction::vt));
		table.insert(std::pair<std::string, ParserInstruction>("vn", ParserInstruction::vn));
	}
	~ObjParser() {}

	enum class ParserInstruction {
		v,
		f,
		vt,
		vn
	};

	virtual bool parse(const std::string& filename, Mesh** _mesh) {
		if (!openFile(filename)) {
			return false;
		}
		std::string lineStr;
		Mesh* mesh = new Mesh;
		int vnIndex = 0;
		std::vector<Vertex*>& vArray = mesh->vArray;
		std::vector<Point2f*>& vtArray = mesh->vtArray;
		std::vector<Vector3f*>& vnArray = mesh->vnArray;
		while (std::getline(moduleFile, lineStr)) {
			std::pair<ParserInstruction, std::vector<float>> lineResult;
			if (lineParse(lineStr, &lineResult)) {
				if (lineResult.first == ParserInstruction::v) {
					Vertex* vertex = new Vertex(Point3f(lineResult.second[0], lineResult.second[1], lineResult.second[2], 1.0f));
					vArray.push_back(vertex);
				}
				else if (lineResult.first == ParserInstruction::f) {
					int v0 = -1;
					int v1 = -1;
					int v2 = -1;
					int v3 = -1;
					int vt0 = -1;
					int vt1 = -1;
					int vt2 = -1;
					int vt3 = -1;
					int vn0 = -1;
					int vn1 = -1;
					int vn2 = -1;
					int vn3 = -1;
					bool isFourVertex = true;
					switch (lineResult.second.size()) {
					case 12:
						// f v/vt/vn v/vt/vn v/vt/vn v/vt/vn
						v0 = lineResult.second[0] - 1;
						v1 = lineResult.second[3] - 1;
						v2 = lineResult.second[6] - 1;
						v3 = lineResult.second[9] - 1;
						vt0 = lineResult.second[1] - 1;
						vt1 = lineResult.second[4] - 1;
						vt2 = lineResult.second[7] - 1;
						vt3 = lineResult.second[10] - 1;
						vn0 = lineResult.second[2] - 1;
						vn1 = lineResult.second[5] - 1;
						vn2 = lineResult.second[8] - 1;
						vn3 = lineResult.second[11] - 1;
						break;
					case 3:
						// f v v v
						isFourVertex = false;
						v0 = lineResult.second[0] - 1;
						v1 = lineResult.second[1] - 1;
						v2 = lineResult.second[2] - 1;
						vtArray.push_back(new Point2f(0, 0));
						vt0 = vt1 = vt2 = 0;
						vnArray.push_back(new Vector3f(computeNormal(vArray[v0]->p, vArray[v1]->p, vArray[v2]->p)));
						vn0 = vn1 = vn2 = 0;
						break;
					case 9:
						// f v/vt/vn v/vt/vn v/vt/vn
						isFourVertex = false;
						v0 = lineResult.second[0] - 1;
						v1 = lineResult.second[3] - 1;
						v2 = lineResult.second[6] - 1;
						vt0 = lineResult.second[1] - 1;
						vt1 = lineResult.second[4] - 1;
						vt2 = lineResult.second[7] - 1;
						vn0 = lineResult.second[2] - 1;
						vn1 = lineResult.second[5] - 1;
						vn2 = lineResult.second[8] - 1;
						break;
					default:
						rlog << "obj f line error\n";
						return false;
					}
					if (isFourVertex) {
						Triple<Vertex*> vTriple0(vArray[v0], vArray[v1], vArray[v2]);
						Triple<Vertex*> vTriple1(vArray[v0], vArray[v2], vArray[v3]);
						Triple<Point2f*> vtTriple0(vtArray[vt0], vtArray[vt1], vtArray[vt2]);
						Triple<Point2f*> vtTriple1(vtArray[vt0], vtArray[vt2], vtArray[vt3]);
						Triple<Vector3f*> vnTriple0(vnArray[vn0], vnArray[vn1], vnArray[vn2]);
						Triple<Vector3f*> vnTriple1(vnArray[vn0], vnArray[vn2], vnArray[vn3]);
						Triangle* tg0 = new Triangle(vTriple0, vtTriple0, vnTriple0);
						Triangle* tg1 = new Triangle(vTriple1, vtTriple1, vnTriple1);
						mesh->add(tg0);
						mesh->add(tg1);
					}
					else {
						Triple<Vertex*> vTriple0(vArray[v0], vArray[v1], vArray[v2]);
						Triple<Point2f*> vtTriple0(vtArray[vt0], vtArray[vt1], vtArray[vt2]);
						Triple<Vector3f*> vnTriple0(vnArray[vn0], vnArray[vn1], vnArray[vn2]);
						Triangle* tg0 = new Triangle(vTriple0, vtTriple0, vnTriple0);
						mesh->add(tg0);
					}
				}
				else if (lineResult.first == ParserInstruction::vn) {
					vnArray.push_back(new Vector3f(lineResult.second[0], lineResult.second[1], lineResult.second[2], 0.0f));
				}
				else if (lineResult.first == ParserInstruction::vt) {
					vtArray.push_back(new Point2f(lineResult.second[0], lineResult.second[1]));
				}
			}
		}
		*_mesh = mesh;
		closeFile();
		return true;
	}

private:
	inline bool isFloatString(const std::string& str) {
		for (const char& le : str) {
			if (le != '-' && le != '.' && le != '0' && le != '1' && le != '2' && le != '3' && le != '4'
				&& le != '5' && le != '6' && le != '7' && le != '8' && le != '9' && le != '+' && le != 'e') {
				return false;
			}
		}
		return true;
	}

	template<typename T>
	inline T convertStringToNumber(const std::string& s) {
		std::istringstream istream(s);
		T r;
		istream >> r;
		return r;
	}

	inline bool lineParse(const std::string& str, std::pair<ParserInstruction, std::vector<float>>* out) {
		std::vector<std::string> e;
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
		}

		return true;
	}

	inline bool vLineParse(const std::vector<std::string>& e, const std::map<std::string, ParserInstruction>::iterator& itor, std::pair<ParserInstruction, std::vector<float>>* out) {
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

	inline bool fLineParse(const std::vector<std::string>& e, const std::map<std::string, ParserInstruction>::iterator& itor, std::pair<ParserInstruction, std::vector<float>>* out) {
		bool first = true;
		int vCount = 0;
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
						&& tl != '5' && tl != '6' && tl != '7' && tl != '8' && tl != '9') {
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
};

