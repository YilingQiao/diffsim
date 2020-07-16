/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
 
#include "collision.hpp"

#include "collisionutil.hpp"
#include "geometry.hpp"
#include "magic.hpp"
#include "optimization.hpp"

#include "timer.hpp"
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <vector>
#include "alglib/linalg.h"
#include "alglib/solvers.h"
#include <torch/torch.h>
#include <algorithm>
#include <utility>
using namespace std;
using namespace alglib;
using torch::Tensor;

#ifndef FAST_MODE

static const int max_iter = 100;
static const Tensor &thickness = ::magic.projection_thickness;
static Tensor obs_mass;
static vector<Tensor> xold;
static vector<Tensor> xold_obs;
static vector<Obstacle*>  obs_obstacles;

namespace CO {

Tensor get_mass (const Node *node) {return is_free(node) ? node->m : obs_mass;}


pair<bool,int> find_in_meshes (const Node *node) {
    int m = find_mesh(node, *::meshes);
    if (m != -1)
        return make_pair(true, m);
    else
        return make_pair(false, find_mesh(node, *::obs_meshes));
}

void update_active (const vector<AccelStruct*> &accs,
                    const vector<AccelStruct*> &obs_accs,
                    const vector<ImpactZone*> &zones);

vector<Impact> find_impacts (const vector<AccelStruct*> &acc,
                             const vector<AccelStruct*> &obs_accs);
vector<Impact> independent_impacts (const vector<Impact> &impacts);

void add_impacts (const vector<Impact> &impacts, vector<ImpactZone*> &zones, const vector<Mesh*> &meshes);

void apply_inelastic_projection (ImpactZone *zone,
                                 const vector<Constraint*> &cons, bool verbose = false);
void update_rigid_trans(int o);


vector<Constraint> impact_constraints (const vector<ImpactZone*> &zones);

ostream &operator<< (ostream &out, const Impact &imp);
ostream &operator<< (ostream &out, const ImpactZone *zone);

void collision_response (Simulation &sim, vector<Mesh*> &meshes, const vector<Constraint*> &cons,
                         const vector<Mesh*> &obs_meshes, bool verbose) {
    ::meshes = &meshes;
    ::obs_meshes = &obs_meshes;
    ::xold = node_positions(meshes);
    ::xold_obs = node_positions(obs_meshes);

    ::obs_obstacles.clear();
    for (int o = 0; o < sim.obstacles.size(); o++) 
        ::obs_obstacles.push_back(&(sim.obstacles[o]));
    

    vector<AccelStruct*> accs = create_accel_structs(meshes, true),
                         obs_accs = create_accel_structs(obs_meshes, true);

    vector<ImpactZone*> zones, prezones;
    ::obs_mass = ONE*1e3;
    int iter;
    static bool changed = false;
    static int count_changed = 0;
    static int num_step = 0;
    num_step++;

    zones.clear();prezones.clear();

    for (iter = 0; iter < max_iter; iter++) {
        zones.clear();
        for (auto p : prezones) {
            ImpactZone *newp = new ImpactZone;
            *newp = *p;
            zones.push_back(newp);
        }

        for (auto p : prezones)
            if (!p->active) 
                delete p;
        if (!zones.empty())
            update_active(accs, obs_accs, zones);

        vector<Impact> impacts = find_impacts(accs, obs_accs);
        impacts = independent_impacts(impacts);
        if (impacts.empty())
            break;

        add_impacts(impacts, zones, obs_meshes);

        for (int z = 0; z < zones.size(); z++) {
            ImpactZone *zone = zones[z];
            if (zone->active){
                changed = true;

                apply_inelastic_projection(zone, cons, verbose);

            }
        }

        for (int o = 0; o < obs_meshes.size(); o++){
            update_rigid_trans(o);
            Tensor v = (obs_meshes[o]->dummy_node->x - obs_meshes[o]->dummy_node->x0) / sim.step_time;
          
        }

        for (int a = 0; a < accs.size(); a++)
            update_accel_struct(*accs[a]);
        for (int a = 0; a < obs_accs.size(); a++)
            update_accel_struct(*obs_accs[a]);

        prezones = zones;
        count_changed++;
    }
 
    if (iter == max_iter) {
        cerr << "Collision resolution failed to converge!" << endl;
        debug_save_meshes(meshes, "meshes");
        debug_save_meshes(obs_meshes, "obsmeshes");
    }
    for (int m = 0; m < meshes.size(); m++) {
        if (changed)
            compute_ws_data(*meshes[m]);
        update_x0(*meshes[m]);
    }

    for (int o = 0; o < obs_meshes.size(); o++) {
        if (changed)
            compute_ws_data(*obs_meshes[o]);

        update_x0(*obs_meshes[o]);

        obs_meshes[o]->dummy_node->v = 
            (obs_meshes[o]->dummy_node->x - obs_meshes[o]->dummy_node->x0) / sim.step_time;
        obs_meshes[o]->dummy_node->x0 = obs_meshes[o]->dummy_node->x;

    }

    ::obs_obstacles.clear();

    for (int z = 0; z < zones.size(); z++)
        delete zones[z];
    destroy_accel_structs(accs);
    destroy_accel_structs(obs_accs);

}


void update_rigid_trans(int o) {

    Node *dummy_node = (::obs_obstacles)[o]->curr_state_mesh.dummy_node; 
    Tensor euler = dummy_node->x.slice(0, 0, 3);
    Tensor trans = dummy_node->x.slice(0, 3, 6);
    Transformation tr;
    tr.rotation    = Quaternion::from_euler(euler);
    tr.translation = trans;
    tr.scale       = dummy_node->scale;
    opt_apply_transformation_onto((::obs_obstacles)[o]->base_mesh, 
        (::obs_obstacles)[o]->curr_state_mesh, tr);

}


void update_active (const vector<AccelStruct*> &accs,
                    const vector<AccelStruct*> &obs_accs,
                    const vector<ImpactZone*> &zones) {
    for (int a = 0; a < accs.size(); a++)
        mark_all_inactive(*accs[a]);

    for (int z = 0; z < zones.size(); z++) {
        const ImpactZone *zone = zones[z];
        if (!zone->active)
            continue;
        for (int n = 0; n < zone->nodes.size(); n++) {
            const Node *node = zone->nodes[n];
            pair<bool,int> mi = find_in_meshes(node);
            AccelStruct *acc = (mi.first ? accs : obs_accs)[mi.second];
            for (int v = 0; v < node->verts.size(); v++)
                for (int f = 0; f < node->verts[v]->adjf.size(); f++)
                    mark_active(*acc, node->verts[v]->adjf[f]);
        }
    }
}


static int nthreads = 0;
static vector<Impact> *impacts = NULL;
static vector<pair<Face const*, Face const*> > *faceimpacts = NULL;
static int *cnt = NULL;

void find_face_impacts (const Face *face0, const Face *face1);

bool vf_collision_test (const Vert *vert, const Face *face, Impact &impact);
bool ee_collision_test (const Edge *edge0, const Edge *edge1, Impact &impact);
bool collision_test (Impact::Type type, const Node *node0, const Node *node1,
                     const Node *node2, const Node *node3, Impact &impact);

void compute_face_impacts (const Face *face0, const Face *face1) {
    int t = omp_get_thread_num();
    Impact impact;
    BOX nb[6], eb[6], fb[2];
    for (int v = 0; v < 3; ++v) {
        nb[v] = node_box(face0->v[v]->node, true);
        nb[v+3] = node_box(face1->v[v]->node, true);
    }
    for (int v = 0; v < 3; ++v) {
        eb[v] = nb[NEXT(v)]+nb[PREV(v)];//edge_box(face0->adje[v], true);//
        eb[v+3] = nb[NEXT(v)+3]+nb[PREV(v)+3];//edge_box(face1->adje[v], true);//
    }
    fb[0] = nb[0]+nb[1]+nb[2];
    fb[1] = nb[3]+nb[4]+nb[5];
    double thick = ::thickness.item<double>();
    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v], fb[1], thick))
            continue;
        if (vf_collision_test(face0->v[v], face1, impact))
            CO::impacts[t].push_back(impact);
    }
    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v+3], fb[0], thick))
            continue;
        if (vf_collision_test(face1->v[v], face0, impact))
            CO::impacts[t].push_back(impact);
    }
    for (int e0 = 0; e0 < 3; e0++)
        for (int e1 = 0; e1 < 3; e1++) {
            if (!overlap(eb[e0], eb[e1+3], thick))
                continue;
            if (ee_collision_test(face0->adje[e0], face1->adje[e1], impact))
                CO::impacts[t].push_back(impact);
        }
}

vector<Impact> find_impacts (const vector<AccelStruct*> &accs,
                             const vector<AccelStruct*> &obs_accs) {
    if (!impacts) {
        CO::nthreads = omp_get_max_threads();
        CO::impacts = new vector<Impact>[CO::nthreads];
        CO::faceimpacts = new vector<pair<Face const*, Face const*> >[CO::nthreads];
        CO::cnt = new int[CO::nthreads];
    }
    for (int t = 0; t < CO::nthreads; t++) {
        CO::impacts[t].clear();
        CO::faceimpacts[t].clear();
        CO::cnt[t] = 0;
    }
    for_overlapping_faces(accs, obs_accs, ::thickness, find_face_impacts);
    vector<pair<Face const*, Face const*> > tot_faces;
    for (int t = 0; t < CO::nthreads; ++t)
        append(tot_faces, CO::faceimpacts[t]);
    #pragma omp parallel for
    for (int i = 0; i < tot_faces.size(); ++i) { 
        compute_face_impacts(tot_faces[i].first,tot_faces[i].second);
       
    }
    vector<Impact> impacts;
    for (int t = 0; t < CO::nthreads; t++) {
        append(impacts, CO::impacts[t]);

    }
    return impacts;
}

void find_face_impacts (const Face *face0, const Face *face1) {
    int t = omp_get_thread_num();
    CO::faceimpacts[t].push_back(make_pair(face0, face1));
}

bool vf_collision_test (const Vert *vert, const Face *face, Impact &impact) {
    const Node *node = vert->node;
    if (node == face->v[0]->node
     || node == face->v[1]->node
     || node == face->v[2]->node)
        return false;
    return collision_test(Impact::VF, node, face->v[0]->node, face->v[1]->node,
                          face->v[2]->node, impact);
}

bool ee_collision_test (const Edge *edge0, const Edge *edge1, Impact &impact) {
    if (edge0->n[0] == edge1->n[0] || edge0->n[0] == edge1->n[1]
        || edge0->n[1] == edge1->n[0] || edge0->n[1] == edge1->n[1])
        return false;
    return collision_test(Impact::EE, edge0->n[0], edge0->n[1],
                          edge1->n[0], edge1->n[1], impact);
}

Tensor pos (const Node *node, Tensor t) {return node->x0 + t*(node->x - node->x0);}


void contact_jacobian(Impact &impact, Node *node) {

    if (is_free(node)) {
       
        impact.imp_Js.push_back(torch::eye(3,TNOPT));
        impact.mesh_num.push_back(-1);
        impact.imp_nodes.push_back(node);

    } else {
        int m = find_mesh(node, *::obs_meshes);
        impact.mesh_num.push_back(m);
        Mesh *mesh = (*::obs_meshes)[m];
        impact.imp_nodes.push_back(mesh->dummy_node);
        Tensor trans = mesh->dummy_node->x.slice(0, 3, 6);
        Tensor euler = mesh->dummy_node->x.slice(0, 0, 3);
        Tensor rt  = node->x - trans;
        Tensor r = (::obs_obstacles)[m]->base_mesh.nodes[node->index]->x;
        Tensor Jw = ZERO33;
        Tensor Jwt = ZERO33;

        Tensor psi=euler[2], theta=euler[1], phi=euler[0];

        Jw[0][2] =-r[0]*cos(theta)*sin(psi)
            + r[1]*(-cos(phi)*cos(psi)-sin(phi)*sin(theta)*sin(psi)) 
            + r[2]*(sin(phi)*cos(psi)-cos(phi)*sin(theta)*sin(psi));

        Jw[0][1] = -r[0]*sin(theta)*cos(psi)
            +r[1]*sin(phi)*cos(theta)*cos(psi)
            +r[2]*cos(phi)*cos(theta)*cos(psi);

        Jw[0][0] = r[1]*sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi)
            +r[2]*(cos(phi)*sin(psi)-sin(phi)*sin(theta)*cos(psi));

        Jw[1][2] = r[0]*cos(theta)*cos(psi)
            +r[1]*(-cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi))
            +r[2]*(sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi));

        Jw[1][1] = -r[0]*sin(theta)*sin(psi) 
            +r[1]*sin(phi)*cos(theta)*sin(psi) 
            +r[2]*cos(phi)*cos(theta)*sin(psi);

        Jw[1][0] = r[1]*(-sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi)) 
            +r[2]*(-cos(phi)*cos(psi)-sin(phi)*sin(theta)*sin(psi));

        Jw[2][2] = ZERO;

        Jw[2][1] = -r[0]*cos(theta)-r[1]*sin(phi)*sin(theta)-r[2]*cos(phi)*sin(theta);

        Jw[2][0] = r[1]*cos(phi)*cos(theta)-r[2]*sin(phi)*cos(theta);

        Jwt[0][1] = rt[2];
        Jwt[0][2] = -rt[1];
        Jwt[1][0] = -rt[2];
        Jwt[1][2] = rt[0];
        Jwt[2][0] = rt[1];
        Jwt[2][1] = -rt[0];

        Tensor J = torch::cat({Jw, torch::eye(3,TNOPT)}, 1);
        impact.imp_Js.push_back(J);
    }
}

bool collision_test (Impact::Type type, const Node *node0, const Node *node1,
                     const Node *node2, const Node *node3, Impact &impact) {
    int t0 = omp_get_thread_num();
        ++CO::cnt[t0];
    impact.type = type;

    impact.imp_nodes.clear();
    impact.imp_Js.clear();    
    impact.mesh_num.clear();

    impact.nodes[0] = (Node*)node0;
    impact.nodes[1] = (Node*)node1;
    impact.nodes[2] = (Node*)node2;
    impact.nodes[3] = (Node*)node3;
    const Tensor &x0 = node0->x0, v0 = node0->x - x0;
    Tensor x1 = node1->x0 - x0, x2 = node2->x0 - x0, x3 = node3->x0 - x0;
    Tensor v1 = (node1->x - node1->x0) - v0, v2 = (node2->x - node2->x0) - v0,
         v3 = (node3->x - node3->x0) - v0;
    Tensor a0 = stp(x1, x2, x3),
           a1 = stp(v1, x2, x3) + stp(x1, v2, x3) + stp(x1, x2, v3),
           a2 = stp(x1, v2, v3) + stp(v1, x2, v3) + stp(v1, v2, x3),
           a3 = stp(v1, v2, v3);
  
    Tensor t = solve_cubic(a3, a2, a1, a0);
    int nsol = t.size(0);

    for (int i = 0; i < nsol; i++) {
        if ((t[i] < 0).item<int>() || (t[i] > 1).item<int>())
            continue;
        impact.t = t[i];
        Tensor bx0 = x0+t[i]*v0, bx1 = x1+t[i]*v1,
             bx2 = x2+t[i]*v2, bx3 = x3+t[i]*v3;
        Tensor &n = impact.n;
        Tensor *w = impact.w;
        w[0] = w[1] = w[2] = w[3] = ZERO;
        Tensor d;
        bool inside, over = false;
        if (type == Impact::VF) {
            d = sub_signed_vf_distance(bx1, bx2, bx3, &n, w, 1e-6, over);
            inside = (torch::min(-w[1], torch::min(-w[2], -w[3])) >= -1e-6).item<int>();
        } else {
            d = sub_signed_ee_distance(bx1, bx2, bx3, bx2-bx1, bx3-bx1, bx3-bx2, &n, w, 1e-6, over);
            inside = (torch::min(torch::min(w[0], w[1]), torch::min(-w[2], -w[3])) >= -1e-6).item<int>();
        }
        if (over || !inside)
            continue;
        if ((dot(n, w[1]*v1 + w[2]*v2 + w[3]*v3) > 0).item<int>())
            n = -n;
       
        contact_jacobian(impact, (Node*)node0);
        contact_jacobian(impact, (Node*)node1);
        contact_jacobian(impact, (Node*)node2);
        contact_jacobian(impact, (Node*)node3);
   
        return true;
    }
    return false;
}


bool operator< (const Impact &impact0, const Impact &impact1) {
    return (impact0.t +0.0001< impact1.t).item<int>();
}

bool conflict (const Impact &impact0, const Impact &impact1);

vector<Impact> independent_impacts (const vector<Impact> &impacts) {
    vector<Impact> sorted = impacts;
    sort(sorted.begin(), sorted.end());
    vector<Impact> indep;

    for (int e = 0; e < sorted.size(); e++) {
        const Impact &impact = sorted[e];
        
        bool con = false;
        for (int e1 = 0; e1 < indep.size(); e1++)
            if (conflict(impact, indep[e1]))
                con = true;
        if (!con)
            indep.push_back(impact);
    }
   
    return indep;
}

bool conflict (const Impact &i0, const Impact &i1) {
    return (i0.imp_nodes[0]->movable && find(i0.nodes[0], i1.nodes)!=-1)
        || (i0.imp_nodes[1]->movable && find(i0.nodes[1], i1.nodes)!=-1)
        || (i0.imp_nodes[2]->movable && find(i0.nodes[2], i1.nodes)!=-1)
        || (i0.imp_nodes[3]->movable && find(i0.nodes[3], i1.nodes)!=-1);
}


ImpactZone *find_or_create_zone (const Node *node, vector<ImpactZone*> &zones);
void merge_zones (ImpactZone* zone0, ImpactZone *zone1,
                  vector<ImpactZone*> &zones);

void add_impacts (const vector<Impact> &impacts, vector<ImpactZone*> &zones, const vector<Mesh*> &meshes) {
    for (int z = 0; z < zones.size(); z++)
        zones[z]->active = false;

    for (int i = 0; i < impacts.size(); i++) {
        const Impact &impact = impacts[i];
      
        Node *node = impact.imp_nodes[impact.imp_nodes[0]->movable ? 0 : 3];

        ImpactZone *zone = find_or_create_zone(node, zones); 

        for (int n = 0; n < 4; n++) {
            if (impact.imp_nodes[n]->movable)
                merge_zones(zone, find_or_create_zone(impact.imp_nodes[n], zones),
                        zones);
        }
        zone->impacts.push_back(impact);
        zone->active = true;
    }
}

ImpactZone *find_or_create_zone (const Node *node, vector<ImpactZone*> &zones) {
    bool is_cloth = true;
    Node *dummy_node = (Node*)node;
    int m = find_mesh_dummy(node, *::obs_meshes);
    if (m != -1) {
        is_cloth  = false;
        dummy_node = (*::obs_meshes)[m]->dummy_node;
    }

    for (int z = 0; z < zones.size(); z++)
        if (is_in(dummy_node, zones[z]->nodes))
            return zones[z];
    ImpactZone *zone = new ImpactZone;
    zone->mesh_num.clear();


    if (is_cloth) {
        zone->mesh_num.push_back(-1);
        zone->nvar = 3;
    } else {
        zone->mesh_num.push_back(m);
        zone->nvar = 6;
    }

    zone->nodes.push_back(dummy_node);
    zones.push_back(zone);

    return zone;
}

void merge_zones (ImpactZone* zone0, ImpactZone *zone1,
                  vector<ImpactZone*> &zones) {
    if (zone0 == zone1)
        return;
    append(zone0->nodes, zone1->nodes);
    append(zone0->impacts, zone1->impacts);

    append(zone0->mesh_num, zone1->mesh_num);

    zone0->nvar += zone1->nvar;

    exclude(zone1, zones);
    delete zone1;
}


struct NormalOpt: public NLConOpt {
    ImpactZone *zone;
    Tensor inv_m;
    vector<double> tmp;
    NormalOpt (): zone(NULL), inv_m(ZERO) {nvar = ncon = 0;}
    NormalOpt (ImpactZone *zone): zone(zone), inv_m(ZERO) {
        nvar = zone->nvar;
        ncon = zone->impacts.size();

        int start_dim = 0;
        zone->node_index.clear();
        for (int n = 0; n < zone->nodes.size(); n++) {
         
            Tensor this_m = (zone->mesh_num[n] == -1) ? zone->nodes[n]->m 
                                : zone->nodes[n]->total_mass;
            inv_m = inv_m + 1/this_m;
         
            zone->node_index.push_back(start_dim);
            start_dim += (zone->mesh_num[n] == -1) ? 3 : 6;
            

        }

        inv_m = inv_m / (double)zone->nodes.size();
        tmp = vector<double>(nvar);
     
    }
    void initialize (double *x) const;
    void precompute (const double *x) const;
    double objective (const double *x) const;
    void obj_grad (const double *x, double *grad) const;
    double constraint (const double *x, int i, int &sign) const;
    void con_grad (const double *x, int i, double factor, double *grad) const;
    void finalize (const double *x);
};

Tensor &get_xold (const Node *node);

void precompute_derivative(real_2d_array &a, real_2d_array &q, real_2d_array &r0, vector<double> &lambda,
                            real_1d_array &sm_1, vector<int> &legals, double **grads, ImpactZone *zone,
                            NormalOpt &slx) {
    a.setlength(slx.nvar,legals.size());
    sm_1.setlength(slx.nvar);
    
    for (int n = 0; n < zone->nodes.size(); n++) {
        const Node *node = zone->nodes[n];
        if (zone->mesh_num[n] == -1) {
            for (int k = 0; k < 3; ++k) 
                sm_1[zone->node_index[n]+k] = 1.0/sqrt(get_mass(node)).item<double>();
        } else {
            for (int k = 0; k < 6; ++k) 
                sm_1[zone->node_index[n]+k] = 1.0/sqrt(node->total_mass).item<double>();  
        }
    } 


    for (int k = 0; k < legals.size(); ++k)
        for (int i = 0; i < slx.nvar; ++i)
            a[i][k]=grads[legals[k]][i] * sm_1[i]; //sqrt(m^-1)
    real_1d_array tau, r1lam1, lamp;
    tau.setlength(slx.nvar);
    
    rmatrixqr(a, slx.nvar, legals.size(), tau);
    real_2d_array qtmp, r, r1;
    int cols = legals.size();
    if (cols>slx.nvar)cols=slx.nvar;
    rmatrixqrunpackq(a, slx.nvar, legals.size(), tau, cols, qtmp);
    rmatrixqrunpackr(a, slx.nvar, legals.size(), r);

    int newdim = 0;
    for (;newdim < cols; ++newdim)
        if (abs(r[newdim][newdim]) < 1e-6)
            break;
    r0.setlength(newdim, newdim);
    r1.setlength(newdim, legals.size() - newdim);
    q.setlength(slx.nvar, newdim);
    for (int i = 0; i < slx.nvar; ++i)
        for (int j = 0; j < newdim; ++j)
            q[i][j] = qtmp[i][j];
    for (int i = 0; i < newdim; ++i) {
        for (int j = 0; j < newdim; ++j)
            r0[i][j] = r[i][j];
        for (int j = newdim; j < legals.size(); ++j)
            r1[i][j-newdim] = r[i][j];
    }
    r1lam1.setlength(newdim);
    for (int i = 0; i < newdim; ++i) {
        r1lam1[i] = 0;
        for (int j = newdim; j < legals.size(); ++j)
            r1lam1[i] += r1[i][j-newdim] * lambda[legals[j]];
    }
    ae_int_t info;
    alglib::densesolverreport rep;
    rmatrixsolve(r0, (ae_int_t)newdim, r1lam1, info, rep, lamp);
    for (int j = 0; j < newdim; ++j)
        lambda[legals[j]] += lamp[j];
    for (int j = newdim; j < legals.size(); ++j)
        lambda[legals[j]] = 0;
    
}

vector<Tensor> apply_inelastic_projection_forward(Tensor xold, Tensor ws, Tensor ns, ImpactZone *zone) {
    
    Timer ti;
    ti.tick();
    auto slx = NormalOpt(zone);
    double x[slx.nvar],oricon[slx.ncon];
    int sign;
    auto lambda = augmented_lagrangian_method(slx);

    
    vector<int> legals;
    double *grads[slx.ncon], tmp;
    
    for (int i = 0; i < slx.ncon; ++i) {
        tmp = slx.constraint(&slx.tmp[0],i,sign);
        grads[i] = NULL;
        if (sign==1 && tmp>1e-6) continue;
        if (sign==-1 && tmp<-1e-6) continue;
        grads[i] = new double[slx.nvar];
        for (int j = 0; j < slx.nvar; ++j)
            grads[i][j]=0;
        slx.con_grad(&slx.tmp[0],i,1,grads[i]);
        legals.push_back(i);
    }
    real_2d_array a, q, r;
    real_1d_array sm_1;
    precompute_derivative(a, q, r, lambda, sm_1, legals, grads, zone, slx);


    Tensor q_tn = arr2ten(q), r_tn = arr2ten(r);
    Tensor lam_tn = ptr2ten(&lambda[0], lambda.size());
    Tensor sm1_tn = ptr2ten(sm_1.getcontent(), sm_1.length());
    Tensor legals_tn = ptr2ten(&legals[0], legals.size());
    Tensor ans = ptr2ten(&slx.tmp[0], slx.nvar);




    for (int i = 0; i < slx.ncon; ++i) {
       delete [] grads[i];
    }
    return {ans.reshape({-1}), q_tn, r_tn, lam_tn, sm1_tn, legals_tn};
}

void apply_inelastic_projection (ImpactZone *zone,
                                 const vector<Constraint*> &cons, bool verbose) {
    py::object func = py::module::import("collision_py").attr("apply_inelastic_projection");
    Tensor inp_xold, inp_w, inp_n;
  
    vector<Tensor> xolds(zone->nodes.size()), ws(zone->impacts.size()*4), ns(zone->impacts.size());
    
    for (int i = 0; i < zone->nodes.size(); ++i) {
        xolds[i] = zone->nodes[i]->xold;
    }
    
    for (int j = 0; j < zone->impacts.size(); ++j) {
        ns[j] = zone->impacts[j].n;
        for (int k = 0; k < 4; ++k)
            ws[j*4+k] = zone->impacts[j].w[k].reshape({1});
    }
    inp_xold = torch::cat(xolds);
    inp_w = torch::cat(ws);
    inp_n = torch::cat(ns);
    double *dw = inp_w.data<double>(), *dn = inp_n.data<double>();
    zone->w = vector<double>(dw, dw+zone->impacts.size()*4);
    zone->n = vector<double>(dn, dn+zone->impacts.size()*3);


    Tensor out_x = func(inp_xold, inp_w, inp_n, zone).cast<Tensor>();

    for (int i = 0; i < zone->nodes.size(); ++i)
        zone->nodes[i]->x = out_x.slice(0, zone->node_index[i],
                             zone->node_index[i] + zone->nodes[i]->x.sizes()[0]);
}

vector<Tensor> compute_derivative(real_1d_array &ans, ImpactZone *zone,
                        real_2d_array &q, real_2d_array &r, real_1d_array &sm_1, vector<int> &legals, 
                        real_1d_array &dldx,
                        vector<double> &lambda, bool verbose=false) {
    real_1d_array qtx, dz, dlam0, dlam, ana, dldw0, dldn0;
    int nvar = zone->nvar;
    int ncon = zone->impacts.size();
    qtx.setlength(q.cols());
    ana.setlength(nvar);
    dldn0.setlength(ncon*3);
    dldw0.setlength(ncon*4);
    dz.setlength(nvar);
    dlam0.setlength(q.cols());
    dlam.setlength(ncon);
    for (int i = 0; i < nvar; ++i)
        ana[i] = dz[i] = 0;
    for (int i = 0; i < ncon*3; ++i) dldn0[i] = 0;
    for (int i = 0; i < ncon*4; ++i) dldw0[i] = 0;

    for (int i = 0; i < q.cols(); ++i) {
        qtx[i] = 0;
        for (int j = 0; j < nvar; ++j) {
            qtx[i] += q[j][i] * dldx[j] * sm_1[j];
        }
   
    }
    // dz = sqrt(m^-1) (sqrt(m^-1) dldx - q * qtx)
    for (int i = 0; i < nvar; ++i) {
        dz[i] = dldx[i] * sm_1[i];
        for (int j = 0; j < q.cols(); ++j)
            dz[i] -= q[i][j] * qtx[j];
        dz[i] *= sm_1[i];
    }
    // dlam = R^-1 * qtx
    ae_int_t info;
    alglib::densesolverreport rep;
    rmatrixsolve(r, (ae_int_t)q.cols(), qtx, info, rep, dlam0);
// cout<<endl;
    for (int j = 0; j < ncon; ++j)
        dlam[j] = 0;
    for (int k = 0; k < q.cols(); ++k)
        dlam[legals[k]] = dlam0[k];
    //part1: dldq * dqdxt = M dz
 
    for (int i = 0; i < nvar; ++i)
        ana[i] += dz[i] / sm_1[i] / sm_1[i];

    //part2: dldg * dgdw * dwdxt
    // for (int j = 0; j < ncon; ++j) {
    //     Impact &imp=zone->impacts[j];
    //     double *dldn = dldn0.getcontent() + j*3;
    //     for (int n = 0; n < 4; n++) {
    //         int i = find(imp.imp_nodes[n], zone->nodes);
    //         double &dldw = dldw0[j*4+n];
    //         if (i != -1) {
    //             for (int k = 0; k < 3; ++k) {
    //                 //g=-w*n*x
    //                 dldw += (dlam[j]*ans[i*3+k]+lambda[j]*dz[i*3+k])*imp.n[k].item<double>();
    // //part3: dldg * dgdn * dndxt
    //                 dldn[k] += imp.w[n].item<double>()*(dlam[j]*ans[i*3+k]+lambda[j]*dz[i*3+k]);
    //             }
    //         } else {
    // //part4: dldh * (dhdw + dhdn)
    //             for (int k = 0; k < 3; ++k) {
    // // TODO
    //                 dldw += (dlam[j] * imp.n[k] * imp.imp_nodes[n]->x[k]).item<double>();
    //                 dldn[k] += (dlam[j] * imp.w[n] * imp.imp_nodes[n]->x[k]).item<double>();
    //             }
    //         }
    //     }
    // }

    // part2: dldg * dgdw * dwdxt
    for (int j = 0; j < ncon; ++j) {
        Impact &imp=zone->impacts[j];
        double *dldn = dldn0.getcontent() + j*3;
        for (int n = 0; n < 4; n++) {
            // auto accessor_J = impact.imp_Js[n].packed_accessor32<double,2>();
            Tensor curnewx = imp.nodes[n]->x;
            double *dx1 = curnewx.data<double>();

            int i = find(imp.imp_nodes[n], zone->nodes);
            double &dldw = dldw0[j*4+n];

            if (zone->mesh_num[i] == -1) {
                for (int k = 0; k < 3; ++k) {
                    //g=-w*n
                    dldw += (dlam[j]*dx1[k]+
                            lambda[j]*dz[zone->node_index[i]+k])*imp.n[k].item<double>();
                    //part3: dldg * dgdn 
                    dldn[k] += imp.w[n].item<double>()*(dlam[j]*ans[zone->node_index[i]+k]+lambda[j]*dz[zone->node_index[i]+k]);
                }
            } else {
                auto accessor_J = imp.imp_Js[n].packed_accessor32<double,2>();
                for (int k = 0; k < 3; k++) {
                    dldw += dlam[j]*dx1[k]*imp.n[k].item<double>();
                    dldn[k] += imp.w[n].item<double>()*dlam[j]*ans[zone->node_index[i]+k] ;
                    for (int q = 0; q < 6; q++) {
                        dldw += lambda[j]*dz[zone->node_index[i]+q]*accessor_J[k][q]*imp.n[k].item<double>();
                        dldn[k] += imp.w[n].item<double>()*lambda[j]*dz[zone->node_index[i]+q]*accessor_J[k][q];
                    }
                }

            }
        }
    }

    Tensor grad_xold = torch::from_blob(ana.getcontent(), {nvar}, TNOPT).clone();
    Tensor grad_w = torch::from_blob(dldw0.getcontent(), {ncon*4}, TNOPT).clone();
    Tensor grad_n = torch::from_blob(dldn0.getcontent(), {ncon* 3}, TNOPT).clone();
    delete zone;
    return {grad_xold, grad_w*0, grad_n*0};
}

vector<Tensor> apply_inelastic_projection_backward(Tensor dldx_tn, Tensor ans_tn, Tensor q_tn, Tensor r_tn, Tensor lam_tn, Tensor sm1_tn, Tensor legals_tn, ImpactZone *zone) {
    real_2d_array q = ten2arr(q_tn), r = ten2arr(r_tn);
    real_1d_array sm_1 = ten1arr(sm1_tn), ans = ten1arr(ans_tn.reshape({-1})), dldx = ten1arr(dldx_tn.reshape({-1}));
    vector<double> lambda = ten2vec<double>(lam_tn);
    vector<int> legals = ten2vec<int>(legals_tn);
    return compute_derivative(ans, zone, q, r, sm_1, legals, dldx, lambda);
}

void NormalOpt::initialize (double *x) const {
    int start_dim = 0;
    for (int n = 0; n < zone->nodes.size(); n++) {
        set_subvec(x, zone->node_index[n], zone->nodes[n]->x);
    }
}

void NormalOpt::precompute (const double *x) const {
    for (int n = 0; n < zone->nodes.size(); n++) {
        bool is_cloth = (zone->mesh_num[n] == -1);
        zone->nodes[n]->x = get_subvec(x, is_cloth, zone->node_index[n]);
        if (!is_cloth) {
            update_rigid_trans(zone->mesh_num[n]);
        }
    }
}

double NormalOpt::objective (const double *x) const {
    double e = 0;
    for (int n = 0; n < zone->nodes.size(); n++) {
        const Node *node = zone->nodes[n];
        Tensor dx = node->x - node->xold;
       
        if (zone->mesh_num[n] == -1) {
            e = e + (inv_m*get_mass(node)*dot(dx, dx)/2).item<double>();
        } else {
            dx = dx.reshape({6,1});
            Tensor w = dx.slice(0,0,3);
            Tensor v = dx.slice(0,3,6);
            Tensor e_ang = inv_m*w.t().matmul(node->ang_inertia).matmul(w).reshape({1});
            Tensor e_lin = inv_m*node->total_mass * v.t().matmul(v).reshape({1});
            e = e + e_ang.item<double>() + e_lin.item<double>();
        }
    }

    return e;
}

void NormalOpt::obj_grad (const double *x, double *grad) const {
    int start_dim = 0;

    for (int n = 0; n < zone->nodes.size(); n++) {
        const Node *node = zone->nodes[n];
        Tensor dx = node->x - node->xold;
        if (zone->mesh_num[n] == -1) {
            set_subvec(grad, start_dim, inv_m*get_mass(node)*dx);
        } else {
            dx = dx.reshape({6,1});
            Tensor w = dx.slice(0,0,3);
            Tensor v = dx.slice(0,3,6);

            Tensor p_ang = inv_m*node->ang_inertia.matmul(w);
            Tensor p_lin = inv_m*node->total_mass * v;

            Tensor v_grad = torch::cat({p_ang, p_lin}, 0).squeeze();
           
            set_subvec(grad, start_dim, v_grad);
        }
        start_dim += zone->nodes[n]->xold.sizes()[0];
    }
}

double NormalOpt::constraint (const double *x, int j, int &sign) const {
    sign = -1;
    double c1 = ::thickness.item<double>();
    const Impact &impact = zone->impacts[j];
    for (int n = 0; n < 4; n++) {
        Tensor curnewx = impact.nodes[n]->x;
        double *dx1 = curnewx.data<double>();
        for (int k = 0; k < 3; ++k) {
            c1 -= zone->w[j*4+n]*zone->n[j*3+k]*dx1[k];
        }
    }
    return c1;
}

void NormalOpt::con_grad (const double *x, int j, double factor,
                          double *grad) const {
    const Impact &impact = zone->impacts[j];
    for (int n = 0; n < 4; n++) {
        Node *node = impact.imp_nodes[n];
        int i = find(node, zone->nodes);

        if (zone->mesh_num[i] == -1) {
            for (int k = 0; k < 3; ++k) {
                grad[zone->node_index[i]+k] -= factor*zone->w[j*4+n]*zone->n[j*3+k];
            }
        } else {
            auto accessor_J = impact.imp_Js[n].packed_accessor32<double,2>();
            for (int q = 0; q < 6; q++) {
                double sum_grad = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum_grad += factor*zone->w[j*4+n]*zone->n[j*3+k]*accessor_J[k][q];
                }
                grad[zone->node_index[i]+q] -= sum_grad;
            }

        }
        
    }
}

void NormalOpt::finalize (const double *x) {
    precompute(x);
    for (int i = 0; i < nvar; ++i) {
        tmp[i] = x[i];
    }
}

Tensor &get_xold (const Node *node) {
    pair<bool,int> mi = find_in_meshes(node);
    int ni = get_index(node, mi.first ? *::meshes : *::obs_meshes);
    return (mi.first ? ::xold : ::xold_obs)[ni];
}

}; 
void collision_response (Simulation &sim, vector<Mesh*> &meshes, const vector<Constraint*> &cons,
                         const vector<Mesh*> &obs_meshes, bool verbose) {
    CO::collision_response(sim, meshes, cons, obs_meshes, verbose);
}

#else


static const int max_iter = 100;
static const Tensor &thickness = ::magic.projection_thickness;
static Tensor obs_mass;
static vector<Tensor> xold;
static vector<Tensor> xold_obs;
static vector<Obstacle*>  obs_obstacles;

namespace CO {

Tensor get_mass (const Node *node) {return is_free(node) ? node->m : obs_mass;}

pair<bool,int> find_in_meshes (const Node *node) {
    int m = find_mesh(node, *::meshes);
    if (m != -1)
        return make_pair(true, m);
    else
        return make_pair(false, find_mesh(node, *::obs_meshes));
}

void update_active (const vector<AccelStruct*> &accs,
                    const vector<AccelStruct*> &obs_accs,
                    const vector<ImpactZone*> &zones);

vector<Impact> find_impacts (const vector<AccelStruct*> &acc,
                             const vector<AccelStruct*> &obs_accs);
vector<Impact> independent_impacts (const vector<Impact> &impacts);

void add_impacts (const vector<Impact> &impacts, vector<ImpactZone*> &zones, vector<Mesh*> &meshes);

void apply_inelastic_projection (ImpactZone *zone,
                                 const vector<Constraint*> &cons, bool verbose = false);
void update_rigid_trans(int o);


vector<Constraint> impact_constraints (const vector<ImpactZone*> &zones);

ostream &operator<< (ostream &out, const Impact &imp);
ostream &operator<< (ostream &out, const ImpactZone *zone);
void t2a_update_rigid_trans(int o);

void collision_response (Simulation &sim, vector<Mesh*> &meshes, const vector<Constraint*> &cons,
                         const vector<Mesh*> &obs_meshes, bool verbose) {
    
    Timer ti;
    ti.tick();

    ::meshes = &meshes;
    ::obs_meshes = &obs_meshes;
    ::xold = node_positions(meshes);
    ::xold_obs = node_positions(obs_meshes);

    ::obs_obstacles.clear();
    for (int o = 0; o < sim.obstacles.size(); o++) 
        ::obs_obstacles.push_back(&(sim.obstacles[o]));
    
    for (int o = 0; o < obs_meshes.size(); o++) {
        Mesh* mesh = obs_meshes[o];
        for (int n = 0; n < mesh->nodes.size(); n++) {
            Node* node = mesh->nodes[n];
            set_subvec(node->d_x0, 0, node->x0);
        }
        set_subvec(mesh->dummy_node->d_x, 0, mesh->dummy_node->x.slice(0, 3, 6));
        set_subvec(mesh->dummy_node->d_angx, 0, mesh->dummy_node->x.slice(0, 0, 3));
        t2a_update_rigid_trans(o);
    } 

    for (int o = 0; o < meshes.size(); o++) {
        Mesh* mesh = meshes[o];
        for (int n = 0; n < mesh->nodes.size(); n++) {
            Node* node = mesh->nodes[n];
            set_subvec(node->d_x, 0, node->x);
            set_subvec(node->d_x0, 0, node->x0);
        }
    }

    vector<AccelStruct*> accs = create_accel_structs(meshes, true),
                         obs_accs = create_accel_structs(obs_meshes, true);

    vector<ImpactZone*> zones, prezones;
    ::obs_mass = ONE*1e3;
    int iter;
    static bool changed = false;
    static int count_changed = 0;
    static int num_step = 0;
    num_step++;
    zones.clear();prezones.clear();

    for (iter = 0; iter < max_iter; iter++) {
        zones.clear();
        for (auto p : prezones) {
            ImpactZone *newp = new ImpactZone;
            *newp = *p;
            zones.push_back(newp);
        }
        for (auto p : prezones)
            if (!p->active) 
                delete p;
        if (!zones.empty())
            update_active(accs, obs_accs, zones);
        
        vector<Impact> impacts = find_impacts(accs, obs_accs);

        impacts = independent_impacts(impacts);
        if (impacts.empty()) 
            break;

        add_impacts(impacts, zones, meshes);

        for (int z = 0; z < zones.size(); z++) {
            ImpactZone *zone = zones[z];
            if (zone->active){
                changed = true;
                apply_inelastic_projection(zone, cons, verbose);
            }
        }

        for (int o = 0; o < obs_meshes.size(); o++){
            update_rigid_trans(o);
            Tensor v = (obs_meshes[o]->dummy_node->x - obs_meshes[o]->dummy_node->x0) / sim.step_time;
        }

        for (int a = 0; a < accs.size(); a++)
            update_accel_struct(*accs[a]);
        for (int a = 0; a < obs_accs.size(); a++)
            update_accel_struct(*obs_accs[a]);

        prezones = zones;
        count_changed++;
    }
     
 
    if (iter == max_iter) {
        cerr << "Collision resolution failed to converge!" << endl;
        debug_save_meshes(meshes, "meshes");
        debug_save_meshes(obs_meshes, "obsmeshes");
    }
     
    for (int m = 0; m < meshes.size(); m++) {
        if (changed)
            compute_ws_data(*meshes[m]);
        update_x0(*meshes[m]);
    }
    
    for (int o = 0; o < obs_meshes.size(); o++) {
        if (changed)
            compute_ws_data(*obs_meshes[o]);

        update_x0(*obs_meshes[o]);
        obs_meshes[o]->dummy_node->v = 
            (obs_meshes[o]->dummy_node->x - obs_meshes[o]->dummy_node->x0) / sim.step_time;
        obs_meshes[o]->dummy_node->x0 = obs_meshes[o]->dummy_node->x;
    }

      
    ::obs_obstacles.clear();

    for (int z = 0; z < zones.size(); z++)
        delete zones[z];
    destroy_accel_structs(accs);
    destroy_accel_structs(obs_accs);
}


void update_rigid_trans(int o) {
    Node *dummy_node = (::obs_obstacles)[o]->curr_state_mesh.dummy_node; 
    Tensor euler = dummy_node->x.slice(0, 0, 3);
    Tensor trans = dummy_node->x.slice(0, 3, 6);
    Transformation tr;
    tr.rotation    = Quaternion::from_euler(euler);
    tr.translation = trans;
    tr.scale       = dummy_node->scale;
    opt_apply_transformation_onto((::obs_obstacles)[o]->base_mesh, 
        (::obs_obstacles)[o]->curr_state_mesh, tr);
}

static void t2a_from_euler(double &s, double *v, const double *euler) {
  double yaw=euler[2], pitch=euler[1], roll=euler[0];

  double cy = std::cos(yaw * 0.5);
  double sy = std::sin(yaw * 0.5);
  double cp = std::cos(pitch * 0.5);
  double sp = std::sin(pitch * 0.5);
  double cr = std::cos(roll * 0.5);
  double sr = std::sin(roll * 0.5);

  s   = cy * cp * cr + sy * sp * sr;
  v[0] = cy * cp * sr - sy * sp * cr;
  v[1] = sy * cp * sr + cy * sp * cr;
  v[2] = sy * cp * cr - cy * sp * sr;
}


static void t2a_trans_node(Node *node_s, Node *node_t, double s, double *v, double *trans) {
  double temp[3], temp2[3];
  double s_temp, s_temp2;
  s_temp = s*s - t2a_dot(v,v);
  t2a_mul_scalar(temp, node_s->d_x, s_temp);
  t2a_mul_scalar(temp2, v, t2a_dot(v,node_s->d_x)*2.0);
  t2a_add(temp, temp, temp2);
  t2a_cross(temp2, v, node_s->d_x);
  t2a_mul_scalar(temp2,  temp2, 2.0*s);
  t2a_add(temp, temp, temp2);
  t2a_add(node_t->d_x, temp, trans);

}

void t2a_update_rigid_trans(int o) {
    double t2a_s, t2a_v[3];

    Node *dummy_node = (::obs_obstacles)[o]->curr_state_mesh.dummy_node; 
    double *euler = dummy_node->d_angx;
    double *translation = dummy_node->d_x;

    t2a_from_euler(t2a_s, t2a_v, euler);

    Mesh &base_mesh = (::obs_obstacles)[o]->base_mesh;  
    Mesh &curr_state_mesh = (::obs_obstacles)[o]->curr_state_mesh;    

    for (int n = 0; n < curr_state_mesh.nodes.size(); n++)
        t2a_trans_node(base_mesh.nodes[n], curr_state_mesh.nodes[n], t2a_s, t2a_v, translation);
}


void update_active (const vector<AccelStruct*> &accs,
                    const vector<AccelStruct*> &obs_accs,
                    const vector<ImpactZone*> &zones) {
    for (int a = 0; a < accs.size(); a++)
        mark_all_inactive(*accs[a]);
    for (int z = 0; z < zones.size(); z++) {
        const ImpactZone *zone = zones[z];
        if (!zone->active)
            continue;
        for (int n = 0; n < zone->nodes.size(); n++) {
            const Node *node = zone->nodes[n];
            pair<bool,int> mi = find_in_meshes(node);
            AccelStruct *acc = (mi.first ? accs : obs_accs)[mi.second];
            for (int v = 0; v < node->verts.size(); v++)
                for (int f = 0; f < node->verts[v]->adjf.size(); f++)
                    mark_active(*acc, node->verts[v]->adjf[f]);
        }
    }
}

static int nthreads = 0;
static vector<Impact> *impacts = NULL;
static vector<pair<Face const*, Face const*> > *faceimpacts = NULL;
static int *cnt = NULL;

void find_face_impacts (const Face *face0, const Face *face1);

bool vf_collision_test (const Vert *vert, const Face *face, Impact &impact);
bool ee_collision_test (const Edge *edge0, const Edge *edge1, Impact &impact);
bool collision_test (Impact::Type type, const Node *node0, const Node *node1,
                     const Node *node2, const Node *node3, Impact &impact);

void compute_face_impacts (const Face *face0, const Face *face1) {
    int t = omp_get_thread_num();
    
    Impact impact;
    BOX nb[6], eb[6], fb[2];
    for (int v = 0; v < 3; ++v) {
        nb[v] = node_box(face0->v[v]->node, true);
        nb[v+3] = node_box(face1->v[v]->node, true);
    }
    for (int v = 0; v < 3; ++v) {
        eb[v] = nb[NEXT(v)]+nb[PREV(v)];
        eb[v+3] = nb[NEXT(v)+3]+nb[PREV(v)+3];
    }
    fb[0] = nb[0]+nb[1]+nb[2];
    fb[1] = nb[3]+nb[4]+nb[5];
    double thick = ::thickness.item<double>();
    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v], fb[1], thick))
            continue;
        if (vf_collision_test(face0->v[v], face1, impact))
            CO::impacts[t].push_back(impact);
    }
    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v+3], fb[0], thick))
            continue;
        if (vf_collision_test(face1->v[v], face0, impact))
            CO::impacts[t].push_back(impact);
    }
    for (int e0 = 0; e0 < 3; e0++)
        for (int e1 = 0; e1 < 3; e1++) {
            if (!overlap(eb[e0], eb[e1+3], thick))
                continue;
            if (ee_collision_test(face0->adje[e0], face1->adje[e1], impact))
                CO::impacts[t].push_back(impact);
        }
}

vector<Impact> find_impacts (const vector<AccelStruct*> &accs,
                             const vector<AccelStruct*> &obs_accs) {
    if (!impacts) {
        CO::nthreads = omp_get_max_threads();
        CO::impacts = new vector<Impact>[CO::nthreads];
        CO::faceimpacts = new vector<pair<Face const*, Face const*> >[CO::nthreads];
        CO::cnt = new int[CO::nthreads];
    }
    for (int t = 0; t < CO::nthreads; t++) {
        CO::impacts[t].clear();
        CO::faceimpacts[t].clear();
        CO::cnt[t] = 0;
    }
    for_overlapping_faces(accs, obs_accs, ::thickness, find_face_impacts);


    vector<pair<Face const*, Face const*> > tot_faces;
    for (int t = 0; t < CO::nthreads; ++t)
        append(tot_faces, CO::faceimpacts[t]);
    #pragma omp parallel for
    for (int i = 0; i < tot_faces.size(); ++i) { 
        compute_face_impacts(tot_faces[i].first,tot_faces[i].second);
       
    }
    vector<Impact> impacts;
    for (int t = 0; t < CO::nthreads; t++) {
        append(impacts, CO::impacts[t]);

    }
    return impacts;
}

void find_face_impacts (const Face *face0, const Face *face1) {
    int t = omp_get_thread_num();
    CO::faceimpacts[t].push_back(make_pair(face0, face1));
}

bool vf_collision_test (const Vert *vert, const Face *face, Impact &impact) {
    const Node *node = vert->node;
    if (node == face->v[0]->node
     || node == face->v[1]->node
     || node == face->v[2]->node)
        return false;
    return collision_test(Impact::VF, node, face->v[0]->node, face->v[1]->node,
                          face->v[2]->node, impact);
}

bool ee_collision_test (const Edge *edge0, const Edge *edge1, Impact &impact) {
    if (edge0->n[0] == edge1->n[0] || edge0->n[0] == edge1->n[1]
        || edge0->n[1] == edge1->n[0] || edge0->n[1] == edge1->n[1])
        return false;
    return collision_test(Impact::EE, edge0->n[0], edge0->n[1],
                          edge1->n[0], edge1->n[1], impact);
}

Tensor pos (const Node *node, Tensor t) {return node->x0 + t*(node->x - node->x0);}

void contact_jacobian(Impact &impact, Node *node, int k) {
    Tensor J;
    if (is_free(node)) {
        J = torch::eye(3,TNOPT);
        impact.mesh_num.push_back(-1);
        impact.imp_nodes.push_back(node);

    } else {
        int m = find_mesh(node, *::obs_meshes);
        impact.mesh_num.push_back(m);
        Mesh *mesh = (*::obs_meshes)[m];
        impact.imp_nodes.push_back(mesh->dummy_node);
        double *trans = mesh->dummy_node->d_x;
        double *euler = mesh->dummy_node->d_angx;
        double rt[3];
        t2a_sub(rt, node->d_x, trans);
        double *r = (::obs_obstacles)[m]->base_mesh.nodes[node->index]->d_x;
        double Jw[3][3];
      
        double psi=euler[2], theta=euler[1], phi=euler[0];

        Jw[0][2] =-r[0]*cos(theta)*sin(psi)
            + r[1]*(-cos(phi)*cos(psi)-sin(phi)*sin(theta)*sin(psi)) 
            + r[2]*(sin(phi)*cos(psi)-cos(phi)*sin(theta)*sin(psi));

        Jw[0][1] = -r[0]*sin(theta)*cos(psi)
            +r[1]*sin(phi)*cos(theta)*cos(psi)
            +r[2]*cos(phi)*cos(theta)*cos(psi);


        Jw[0][0] = r[1]*sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi)
            +r[2]*(cos(phi)*sin(psi)-sin(phi)*sin(theta)*cos(psi));

        Jw[1][2] = r[0]*cos(theta)*cos(psi)
            +r[1]*(-cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi))
            +r[2]*(sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi));

        Jw[1][1] = -r[0]*sin(theta)*sin(psi) 
            +r[1]*sin(phi)*cos(theta)*sin(psi) 
            +r[2]*cos(phi)*cos(theta)*sin(psi);

        Jw[1][0] = r[1]*(-sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi)) 
            +r[2]*(-cos(phi)*cos(psi)-sin(phi)*sin(theta)*sin(psi));

        Jw[2][2] = 0.0;

        Jw[2][1] = -r[0]*cos(theta)-r[1]*sin(phi)*sin(theta)-r[2]*cos(phi)*sin(theta);

        Jw[2][0] = r[1]*cos(phi)*cos(theta)-r[2]*sin(phi)*cos(theta);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                impact.d_Js[k][i][j] = Jw[i][j]; 
            for (int j = 0; j < 3; j++) {
                if (i!=j)
                    impact.d_Js[k][i][j+3] = 0.0;
                else
                    impact.d_Js[k][i][j+3] = 1.0;
            }
        }
    }
  
}


static double t2a_newtons_method (double a, double b, double c, double d, double x0,
                       int init_dir) {
    if (init_dir != 0) {
        double y0 = d + x0*(c + x0*(b + x0*a)),
               ddy0 = 2*b + x0*(6*a);
        x0 += init_dir*sqrt(abs(2*y0/ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        double y = d + x0*(c + x0*(b + x0*a));
        double dy = c + x0*(2*b + x0*3*a);
        if (dy == 0)
            return x0;
        double x1 = x0 - y/dy;
        if (abs(x0 - x1) < 1e-6)
            return x0;
        x0 = x1;
    }
    return x0;
}

template <typename T> T sgn (const T &x) {return x<0 ? -1 : 1;}

static int t2a_solve_quadratic (double a, double b, double c, double x[2]) {
    // http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
    double d = b*b - 4*a*c;
    if (d < 0) {
        x[0] = -b/(2*a);
        return 0;
    }
    double q = -(b + sgn(b)*sqrt(d))/2;
    int i = 0;
    if (abs(a) > 1e-12*abs(q))
        x[i++] = q/a;
    if (abs(q) > 1e-12*abs(c))
        x[i++] = c/q;
    if (i==2 && x[0] > x[1])
        swap(x[0], x[1]);
    return i;
}


static int t2a_solve_cubic (double a, double b, double c, double d, double x[3]) {
    double xc[2];
    int ncrit = t2a_solve_quadratic(3*a, 2*b, c, xc);
    if (ncrit == 0) {
        x[0] = t2a_newtons_method(a, b, c, d, xc[0], 0);
        return 1;
    } else if (ncrit == 1) {
        return t2a_solve_quadratic(b, c, d, x);
    } else {
        double yc[2] = {d + xc[0]*(c + xc[0]*(b + xc[0]*a)),
                        d + xc[1]*(c + xc[1]*(b + xc[1]*a))};
        int i = 0;
        if (yc[0]*a >= 0)
            x[i++] = t2a_newtons_method(a, b, c, d, xc[0], -1);
        if (yc[0]*yc[1] <= 0) {
            int closer = abs(yc[0])<abs(yc[1]) ? 0 : 1;
            x[i++] = t2a_newtons_method(a, b, c, d, xc[closer], closer==0?1:-1);
        }
        if (yc[1]*a <= 0)
            x[i++] = t2a_newtons_method(a, b, c, d, xc[1], 1);
        return i;
    }
}


static double t2a_sub_signed_ee_distance( double* x1mx0,  double* y0mx0,  double* y1mx0,
                            double* y0mx1,  double* y1mx1,  double* y1my0,
                           double *n, double *w, double thres, bool &over) {
    t2a_cross(n, x1mx0, y1my0);
    w[0]=w[1]=w[2]=w[3]=0;
    if (t2a_norm2(n) < 1e-6) {
        over = true;
        return std::numeric_limits<double>::infinity();
    }
    t2a_normalize(n);
    double h = -t2a_dot(y0mx0, n);
    over = (abs(h) > thres);
      if (over) return h;

    double a0 = t2a_stp(y1mx1, y0mx1, n), a1 = t2a_stp(y0mx0, y1mx0, n),
           b0 = t2a_stp(y1mx1, y1mx0, n), b1 = t2a_stp(y0mx0, y0mx1, n);
    double suma = 1/(a0+a1), sumb = 1/(b0+b1);
    w[0] = a0*suma;
    w[1] = a1*suma;
    w[2] = -b0*sumb;
    w[3] = -b1*sumb;
    return h;
}

static double t2a_sub_signed_vf_distance( double *y0,  double *y1,  double *y2,
                           double *n, double *w, double thres, bool &over) {
    double temp1[3], temp2[3];
    t2a_sub(temp1, y1, y0);
    t2a_sub(temp2, y2, y0);
    t2a_cross(n, temp1, temp2);
    w[0]=w[1]=w[2]=w[3]=0;
    if (t2a_norm2(n) < 1e-6) {
        over = true;
        return std::numeric_limits<double>::infinity();
    } 
    t2a_normalize(n);
    double h = -t2a_dot(y0, n);
    over = (abs(h) > thres);
    if (over) return h;
    double b0 = t2a_stp(y1, y2, n),
           b1 = t2a_stp(y2, y0, n),
           b2 = t2a_stp(y0, y1, n);
    double sum = 1/(b0 + b1 + b2);
    w[0] = 1;
    w[1] = -b0*sum;
    w[2] = -b1*sum;
    w[3] = -b2*sum;
    return h;
}

bool collision_test (Impact::Type type, const Node *node0, const Node *node1,
                     const Node *node2, const Node *node3, Impact &impact) {
    int t0 = omp_get_thread_num();
        ++CO::cnt[t0];
    impact.type = type;

    impact.imp_nodes.clear();
    impact.imp_Js.clear();    
    impact.mesh_num.clear();

    impact.nodes[0] = (Node*)node0;
    impact.nodes[1] = (Node*)node1;
    impact.nodes[2] = (Node*)node2;
    impact.nodes[3] = (Node*)node3;

    const double* x0 = node0->d_x0;
    double v0[3], x1[3], x2[3], x3[3];
    double v1[3], v2[3], v3[3], temp[3];
    double a0, a1, a2, a3;

    t2a_sub(v0, node0->d_x, x0);
    t2a_sub(x1, node1->d_x0, x0);
    t2a_sub(x2, node2->d_x0, x0);
    t2a_sub(x3, node3->d_x0, x0);

    t2a_sub(temp, node1->d_x, node1->d_x0);
    t2a_sub(v1, temp, v0);
    t2a_sub(temp, node2->d_x, node2->d_x0);
    t2a_sub(v2, temp, v0);
    t2a_sub(temp, node3->d_x, node3->d_x0);
    t2a_sub(v3, temp, v0);

    a0 = t2a_stp(x1, x2, x3);
    a1 = t2a_stp(v1, x2, x3) + t2a_stp(x1, v2, x3) + t2a_stp(x1, x2, v3);
    a2 = t2a_stp(x1, v2, v3) + t2a_stp(v1, x2, v3) + t2a_stp(v1, v2, x3);
    a3 = t2a_stp(v1, v2, v3);

    
    double d_t[4];
    int nsol = t2a_solve_cubic(a3, a2, a1, a0, d_t);
    d_t[nsol] = 1;

    for (int i = 0; i < nsol; i++) {
        if ((d_t[i] < 0) || (d_t[i] > 1))
            continue;
        impact.t = ONE * d_t[i];
        double bx0[3], bx1[3], bx2[3], bx3[3], temp[3], temp1[3], temp2[3],temp_sum[3];
        t2a_mul_scalar(bx0, v0, d_t[i]); t2a_add(bx0, bx0, x0);
        t2a_mul_scalar(bx1, v1, d_t[i]); t2a_add(bx1, bx1, x1);
        t2a_mul_scalar(bx2, v2, d_t[i]); t2a_add(bx2, bx2, x2);
        t2a_mul_scalar(bx3, v3, d_t[i]); t2a_add(bx3, bx3, x3);

        
        double n[3], w[4];

        w[0] = w[1] = w[2] = w[3] = 0;
        double d;
        bool inside, over = false;
        if (type == Impact::VF) {
            d = t2a_sub_signed_vf_distance(bx1, bx2, bx3, n, w, 1e-6, over);
            inside = std::min(-w[1], std::min(-w[2], -w[3])) >= -1e-6;
        } else {// Impact::EE
            t2a_sub(temp, bx2, bx1);
            t2a_sub(temp1, bx3, bx1);
            t2a_sub(temp2, bx3, bx2);
            d = t2a_sub_signed_ee_distance(bx1, bx2, bx3, temp, temp1, temp2, n, w, 1e-6, over);
            inside = std::min(std::min(w[0], w[1]), std::min(-w[2], -w[3])) >= -1e-6;
        }
        if (over || !inside)
            continue;

        t2a_mul_scalar(temp, v1, w[1]);
        t2a_mul_scalar(temp_sum, v2, w[2]);
        t2a_add(temp_sum, temp_sum, temp);
        t2a_mul_scalar(temp, v3, w[3]);
        t2a_add(temp_sum, temp_sum, temp);

       
        if (t2a_dot(n, temp_sum) > 0)
            t2a_mul_scalar(n, n, -1);

        contact_jacobian(impact, (Node*)node0, 0);
        contact_jacobian(impact, (Node*)node1, 1);
        contact_jacobian(impact, (Node*)node2, 2);
        contact_jacobian(impact, (Node*)node3, 3);


        impact.n = torch::tensor(vector<double>(n,n+3), TNOPT);
        impact.w[0] = torch::tensor(vector<double>(w,w+1), TNOPT);
        impact.w[1] = torch::tensor(vector<double>(w+1,w+2), TNOPT);
        impact.w[2] = torch::tensor(vector<double>(w+2,w+3), TNOPT);
        impact.w[3] = torch::tensor(vector<double>(w+3,w+4), TNOPT);
        return true;

    }
    return false;
}


bool operator< (const Impact &impact0, const Impact &impact1) {
    return (impact0.t +0.0001< impact1.t).item<int>();
}

bool conflict (const Impact &impact0, const Impact &impact1);

vector<Impact> independent_impacts (const vector<Impact> &impacts) {
    vector<Impact> sorted = impacts;
    sort(sorted.begin(), sorted.end());
    vector<Impact> indep;
    for (int e = 0; e < sorted.size(); e++) {
        const Impact &impact = sorted[e];
 
        bool con = false;
        for (int e1 = 0; e1 < indep.size(); e1++)
            if (conflict(impact, indep[e1]))
                con = true;
        if (!con)
            indep.push_back(impact);
    }
    return indep;
}

bool conflict (const Impact &i0, const Impact &i1) {
    return (i0.imp_nodes[0]->movable && find(i0.nodes[0], i1.nodes)!=-1)
        || (i0.imp_nodes[1]->movable && find(i0.nodes[1], i1.nodes)!=-1)
        || (i0.imp_nodes[2]->movable && find(i0.nodes[2], i1.nodes)!=-1)
        || (i0.imp_nodes[3]->movable && find(i0.nodes[3], i1.nodes)!=-1);
}

// Impact zones

ImpactZone *find_or_create_zone (const Node *node, vector<ImpactZone*> &zones);
void merge_zones (ImpactZone* zone0, ImpactZone *zone1,
                  vector<ImpactZone*> &zones);

void add_impacts (const vector<Impact> &impacts, vector<ImpactZone*> &zones, vector<Mesh*> &meshes) {
    for (int z = 0; z < zones.size(); z++)
        zones[z]->active = false;

    for (int i = 0; i < impacts.size(); i++) {
        const Impact &impact = impacts[i];
        Node *node = impact.imp_nodes[impact.imp_nodes[0]->movable ? 0 : 3];
        ImpactZone *zone = find_or_create_zone(node, zones); 
        for (int n = 0; n < 4; n++) {
            if (impact.imp_nodes[n]->movable)
                merge_zones(zone, find_or_create_zone(impact.imp_nodes[n], zones),
                        zones);
        }
        zone->impacts.push_back(impact);
        zone->active = true;
    }
}

ImpactZone *find_or_create_zone (const Node *node, vector<ImpactZone*> &zones) {
    bool is_cloth = true;
    Node *dummy_node = (Node*)node;
    int m = find_mesh_dummy(node, *::obs_meshes);
    if (m != -1) {
        is_cloth  = false;
        dummy_node = (*::obs_meshes)[m]->dummy_node;
    }
    for (int z = 0; z < zones.size(); z++)
        if (is_in(dummy_node, zones[z]->nodes))
            return zones[z];
    ImpactZone *zone = new ImpactZone;
    zone->mesh_num.clear();
    if (is_cloth) {
        zone->mesh_num.push_back(-1);
        zone->nvar = 3;
    } else {
        zone->mesh_num.push_back(m);
        zone->nvar = 6;
    }

    zone->nodes.push_back(dummy_node);
    zones.push_back(zone);

    return zone;
}

void merge_zones (ImpactZone* zone0, ImpactZone *zone1,
                  vector<ImpactZone*> &zones) {
    if (zone0 == zone1)
        return;
    append(zone0->nodes, zone1->nodes);
    append(zone0->impacts, zone1->impacts);
    append(zone0->mesh_num, zone1->mesh_num);
    zone0->nvar += zone1->nvar;
    exclude(zone1, zones);
    delete zone1;
}

// Response
struct NormalOpt: public NLConOpt {
    ImpactZone *zone;
    Tensor inv_m;
    vector<double> tmp;
    double ini_c = ::thickness.item<double>();
    double ini_c1 = ::thickness.item<double>();
    double t2a_invm;

    NormalOpt (): zone(NULL), inv_m(ZERO) {nvar = ncon = 0;}
    NormalOpt (ImpactZone *zone): zone(zone), inv_m(ZERO) {
        nvar = zone->nvar;
        ncon = zone->impacts.size();
        int start_dim = 0;
        zone->node_index.clear();
        for (int n = 0; n < zone->nodes.size(); n++) {
            Tensor this_m = (zone->mesh_num[n] == -1) ? zone->nodes[n]->m 
                                : zone->nodes[n]->total_mass;
            inv_m = inv_m + 1/this_m;
            zone->node_index.push_back(start_dim);
            start_dim += (zone->mesh_num[n] == -1) ? 3 : 6;
        }
        inv_m = inv_m / (double)zone->nodes.size();
        tmp = vector<double>(nvar);
        t2a_invm = inv_m.item<double>();
    }
    void initialize (double *x) const;
    void precompute (const double *x) const;
    double objective (const double *x) const;
    void obj_grad (const double *x, double *grad) const;
    double constraint (const double *x, int i, int &sign) const;
    void con_grad (const double *x, int i, double factor, double *grad) const;
    void finalize (const double *x);
};

Tensor &get_xold (const Node *node);

void precompute_derivative(real_2d_array &a, real_2d_array &q, real_2d_array &r0, vector<double> &lambda,
                            real_1d_array &sm_1, vector<int> &legals, double **grads, ImpactZone *zone,
                            NormalOpt &slx) {
    a.setlength(slx.nvar,legals.size());
    sm_1.setlength(slx.nvar);
    for (int n = 0; n < zone->nodes.size(); n++) {
        const Node *node = zone->nodes[n];
        if (zone->mesh_num[n] == -1) {
            for (int k = 0; k < 3; ++k) 
                sm_1[zone->node_index[n]+k] = 1.0/sqrt(get_mass(node)).item<double>();
        } else {
            for (int k = 0; k < 6; ++k) 
                sm_1[zone->node_index[n]+k] = 1.0/sqrt(node->total_mass).item<double>();  
        }
    } 


    for (int k = 0; k < legals.size(); ++k)
        for (int i = 0; i < slx.nvar; ++i)
            a[i][k]=grads[legals[k]][i] * sm_1[i]; //sqrt(m^-1)
    real_1d_array tau, r1lam1, lamp;
    tau.setlength(slx.nvar);
    
    rmatrixqr(a, slx.nvar, legals.size(), tau);
    real_2d_array qtmp, r, r1;
    int cols = legals.size();
    if (cols>slx.nvar)cols=slx.nvar;
    rmatrixqrunpackq(a, slx.nvar, legals.size(), tau, cols, qtmp);
    rmatrixqrunpackr(a, slx.nvar, legals.size(), r);

    int newdim = 0;
    for (;newdim < cols; ++newdim)
        if (abs(r[newdim][newdim]) < 1e-6)
            break;
    r0.setlength(newdim, newdim);
    r1.setlength(newdim, legals.size() - newdim);
    q.setlength(slx.nvar, newdim);
    for (int i = 0; i < slx.nvar; ++i)
        for (int j = 0; j < newdim; ++j)
            q[i][j] = qtmp[i][j];
    for (int i = 0; i < newdim; ++i) {
        for (int j = 0; j < newdim; ++j)
            r0[i][j] = r[i][j];
        for (int j = newdim; j < legals.size(); ++j)
            r1[i][j-newdim] = r[i][j];
    }
    r1lam1.setlength(newdim);
    for (int i = 0; i < newdim; ++i) {
        r1lam1[i] = 0;
        for (int j = newdim; j < legals.size(); ++j)
            r1lam1[i] += r1[i][j-newdim] * lambda[legals[j]];
    }
    ae_int_t info;
    alglib::densesolverreport rep;
    rmatrixsolve(r0, (ae_int_t)newdim, r1lam1, info, rep, lamp);
    for (int j = 0; j < newdim; ++j)
        lambda[legals[j]] += lamp[j];
    for (int j = newdim; j < legals.size(); ++j)
        lambda[legals[j]] = 0;
    
}

vector<Tensor> apply_inelastic_projection_forward(Tensor xold, Tensor ws, Tensor ns, ImpactZone *zone) {
    
    Timer ti;
    ti.tick();

    auto slx = NormalOpt(zone);
    double x[slx.nvar],oricon[slx.ncon];
    int sign;
    auto lambda = augmented_lagrangian_method(slx);



    vector<int> legals;
    double *grads[slx.ncon], tmp;
    
    for (int i = 0; i < slx.ncon; ++i) {
        tmp = slx.constraint(&slx.tmp[0],i,sign);
        grads[i] = NULL;
        if (sign==1 && tmp>1e-6) continue;//sign==1:tmp>=0
        if (sign==-1 && tmp<-1e-6) continue;
        grads[i] = new double[slx.nvar];
        for (int j = 0; j < slx.nvar; ++j)
            grads[i][j]=0;
        slx.con_grad(&slx.tmp[0],i,1,grads[i]);
        legals.push_back(i);
    }
    real_2d_array a, q, r;
    real_1d_array sm_1;//sqrt(m^-1)
    precompute_derivative(a, q, r, lambda, sm_1, legals, grads, zone, slx);

    ti.tock();

    Tensor q_tn = arr2ten(q), r_tn = arr2ten(r);
    Tensor lam_tn = ptr2ten(&lambda[0], lambda.size());
    Tensor sm1_tn = ptr2ten(sm_1.getcontent(), sm_1.length());
    Tensor legals_tn = ptr2ten(&legals[0], legals.size());
    Tensor ans = ptr2ten(&slx.tmp[0], slx.nvar);
    for (int i = 0; i < slx.ncon; ++i) {
       delete [] grads[i];
    }
  
    return {ans.reshape({-1}), q_tn, r_tn, lam_tn, sm1_tn, legals_tn};
}

void apply_inelastic_projection (ImpactZone *zone,
                                 const vector<Constraint*> &cons, bool verbose) {
    py::object func = py::module::import("collision_py").attr("apply_inelastic_projection");
    Tensor inp_xold, inp_w, inp_n;
    vector<Tensor> xolds(zone->nodes.size()), ws(zone->impacts.size()*4), ns(zone->impacts.size());

    for (int i = 0; i < zone->nodes.size(); ++i) {

        xolds[i] = zone->nodes[i]->xold;
    }
    
    for (int j = 0; j < zone->impacts.size(); ++j) {
        ns[j] = zone->impacts[j].n;
        for (int k = 0; k < 4; ++k)
            ws[j*4+k] = zone->impacts[j].w[k].reshape({1});
    }
    inp_xold = torch::cat(xolds);
    inp_w = torch::cat(ws);
    inp_n = torch::cat(ns);
    double *dw = inp_w.data<double>(), *dn = inp_n.data<double>();
    zone->w = vector<double>(dw, dw+zone->impacts.size()*4);
    zone->n = vector<double>(dn, dn+zone->impacts.size()*3);
    Tensor out_x = func(inp_xold, inp_w, inp_n, zone).cast<Tensor>();

    for (int i = 0; i < zone->nodes.size(); ++i)
        zone->nodes[i]->x = out_x.slice(0, zone->node_index[i],
                             zone->node_index[i] + zone->nodes[i]->x.sizes()[0]);
}

vector<Tensor> compute_derivative(real_1d_array &ans, ImpactZone *zone,
                        real_2d_array &q, real_2d_array &r, real_1d_array &sm_1, vector<int> &legals, 
                        real_1d_array &dldx,
                        vector<double> &lambda, bool verbose=false) {
    
    real_1d_array qtx, dz, dlam0, dlam, ana, dldw0, dldn0;
    int nvar = zone->nvar;
    int ncon = zone->impacts.size();
    qtx.setlength(q.cols());
    ana.setlength(nvar);
    dldn0.setlength(ncon*3);
    dldw0.setlength(ncon*4);
    dz.setlength(nvar);
    dlam0.setlength(q.cols());
    dlam.setlength(ncon);
    for (int i = 0; i < nvar; ++i)
        ana[i] = dz[i] = 0;
    for (int i = 0; i < ncon*3; ++i) dldn0[i] = 0;
    for (int i = 0; i < ncon*4; ++i) dldw0[i] = 0;
    // qtx = qt * sqrt(m^-1) dldx
    for (int i = 0; i < q.cols(); ++i) {
        qtx[i] = 0;
        for (int j = 0; j < nvar; ++j) {
            qtx[i] += q[j][i] * dldx[j] * sm_1[j];
        }
    }
    // dz = sqrt(m^-1) (sqrt(m^-1) dldx - q * qtx)
    for (int i = 0; i < nvar; ++i) {
        dz[i] = dldx[i] * sm_1[i];
        for (int j = 0; j < q.cols(); ++j)
            dz[i] -= q[i][j] * qtx[j];
        dz[i] *= sm_1[i];
    }
    //part1: dldq * dqdxt = M dz
    for (int i = 0; i < nvar; ++i)
        ana[i] += dz[i] / sm_1[i] / sm_1[i];

    Tensor grad_xold = torch::from_blob(ana.getcontent(), {nvar}, TNOPT).clone();
    Tensor grad_w = torch::from_blob(dldw0.getcontent(), {ncon*4}, TNOPT).clone();
    Tensor grad_n = torch::from_blob(dldn0.getcontent(), {ncon* 3}, TNOPT).clone();
    delete zone;

    return {grad_xold, grad_w*0, grad_n*0};
}

vector<Tensor> apply_inelastic_projection_backward(Tensor dldx_tn, Tensor ans_tn, Tensor q_tn, Tensor r_tn, Tensor lam_tn, Tensor sm1_tn, Tensor legals_tn, ImpactZone *zone) {
    real_2d_array q = ten2arr(q_tn), r = ten2arr(r_tn);
    real_1d_array sm_1 = ten1arr(sm1_tn), ans = ten1arr(ans_tn.reshape({-1})), dldx = ten1arr(dldx_tn.reshape({-1}));
    vector<double> lambda = ten2vec<double>(lam_tn);
    vector<int> legals = ten2vec<int>(legals_tn);
    return compute_derivative(ans, zone, q, r, sm_1, legals, dldx, lambda);
}



void NormalOpt::initialize (double *x) const {
    int start_dim = 0;
    for (int n = 0; n < zone->nodes.size(); n++) {
        set_subvec(x, zone->node_index[n], zone->nodes[n]->x);
        Node *node = zone->nodes[n];
        if (zone->mesh_num[n] == -1) {
            set_subvec(node->d_x, 0, node->x);
            set_subvec(node->d_xold, 0, node->xold);
            node->d_mass[0] = node->m.item<double>();

        } else {
            set_subvec(node->d_x, 0, node->x.slice(0, 3, 6));
            set_subvec(node->d_xold, 0, node->xold.slice(0, 3, 6));
            node->d_mass[0] = node->total_mass.item<double>();

            set_subvec(node->d_angx, 0, node->x.slice(0, 0, 3));
            set_subvec(node->d_angxold, 0, node->xold.slice(0, 0, 3));

            auto accessor_ang_inertia = node->ang_inertia.packed_accessor<double,2>();
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) 
                    node->d_ang_inertia[i][j] = accessor_ang_inertia[i][j];
        }
    }

}

void NormalOpt::precompute (const double *x) const {
    for (int n = 0; n < zone->nodes.size(); n++) {
        if (zone->mesh_num[n] == -1) {
            t2a_get_subvec(zone->nodes[n]->d_x, x, zone->node_index[n]);
        } else {
            Node* node = zone->nodes[n];
            t2a_get_subvec(zone->nodes[n]->d_angx, x, zone->node_index[n]);
            t2a_get_subvec(zone->nodes[n]->d_x, x, zone->node_index[n]+3);
            t2a_update_rigid_trans(zone->mesh_num[n]);
        }
    }
}

double NormalOpt::objective (const double *x) const {
    double e = 0;
    double temp[3], temp2[3];

    for (int n = 0; n < zone->nodes.size(); n++) {

        const Node *node = zone->nodes[n];

        if (zone->mesh_num[n] == -1) {
            t2a_sub(temp, node->d_x, node->d_xold);
            double e_lin = t2a_dot(temp, temp)* node->d_mass[0] * t2a_invm/2.0;
            e = e + e_lin ;
        } else {
            t2a_sub(temp, node->d_x, node->d_xold);
            double e_lin = t2a_dot(temp, temp)* node->d_mass[0] * t2a_invm;
            t2a_sub(temp, node->d_angx, node->d_angxold);
            t2a_vec_mul_matrix(temp2, temp, node->d_ang_inertia);
            double e_ang = t2a_dot(temp, temp2) * t2a_invm;
            e = e + e_lin + e_ang;
        }
    }
    return e;
}

void NormalOpt::obj_grad (const double *x, double *grad) const {

    double temp[3], temp2[3];
    int start_dim = 0;
    for (int n = 0; n < zone->nodes.size(); n++) {
        const Node *node = zone->nodes[n];
        if (zone->mesh_num[n] == -1) {
            t2a_sub(temp, node->d_x, node->d_xold);
            t2a_mul_scalar(temp, temp, node->d_mass[0]*t2a_invm);
            t2a_set_subvec(grad, start_dim, temp);
            start_dim += 3;
        } else {
            t2a_sub(temp, node->d_angx, node->d_angxold);
            t2a_matrix_mul_vec(temp2, node->d_ang_inertia, temp);
            t2a_mul_scalar(temp2, temp2, t2a_invm);
            t2a_set_subvec(grad, start_dim, temp2);
            t2a_sub(temp, node->d_x, node->d_xold);
            t2a_mul_scalar(temp, temp, t2a_invm * node->d_mass[0]);
            t2a_set_subvec(grad, start_dim+3, temp);
            start_dim += 6;
        }
    }

}

double NormalOpt::constraint (const double *x, int j, int &sign) const {
    sign = -1;
    const Impact &impact = zone->impacts[j];
    double c1 = ini_c1;
    for (int n = 0; n < 4; n++) {
        double *dx1 = impact.nodes[n]->d_x;
        for (int k = 0; k < 3; ++k) {
            c1 -= zone->w[j*4+n]*zone->n[j*3+k]*dx1[k];
        }
    }
    return c1;
}

void NormalOpt::con_grad (const double *x, int j, double factor,
                          double *grad) const {
    const Impact &impact = zone->impacts[j];
    for (int n = 0; n < 4; n++) {
        Node *node = impact.imp_nodes[n];
        int i = find(node, zone->nodes);

        if (i != -1) {
            if (zone->mesh_num[i] == -1) {
                for (int k = 0; k < 3; ++k) {
                    grad[zone->node_index[i]+k] -= factor*zone->w[j*4+n]*zone->n[j*3+k];  
                }
            } else {
                for (int q = 0; q < 6; q++) {
                    double sum_grad = 0.0;
                    for (int k = 0; k < 3; k++) {
                        sum_grad += factor*zone->w[j*4+n]*zone->n[j*3+k]*impact.d_Js[n][k][q];
                    }
                    grad[zone->node_index[i]+q] -= sum_grad;
                }
            }
        }      
    }
}

void NormalOpt::finalize (const double *x) {
    precompute(x);
    for (int i = 0; i < nvar; ++i) {
        tmp[i] = x[i];
    }
}

Tensor &get_xold (const Node *node) {
    pair<bool,int> mi = find_in_meshes(node);
    int ni = get_index(node, mi.first ? *::meshes : *::obs_meshes);
    return (mi.first ? ::xold : ::xold_obs)[ni];
}

}; //namespace CO

void collision_response (Simulation &sim, vector<Mesh*> &meshes, const vector<Constraint*> &cons,
                         const vector<Mesh*> &obs_meshes, bool verbose) {
    CO::collision_response(sim, meshes, cons, obs_meshes, verbose);
}

#endif