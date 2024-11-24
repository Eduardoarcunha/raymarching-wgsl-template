fn sdf_round_box(p: vec3f, b: vec3f, r: f32, quat: vec4f) -> f32
{
    // Apply rotation using quaternion
    var rotated_p = rotate_vector(p, quat);
    
    // Calculate the distance
    var d = abs(rotated_p) - b;
    
    // Return minimum of component-wise max with 0 and length of max of d with 0
    return length(max(d, vec3f(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0) - r;
}

fn sdf_sphere(p: vec3f, r: vec4f, quat: vec4f) -> f32
{
    // Apply rotation using quaternion if needed
    var rotated_p = rotate_vector(p, quat);
    
    // Scale the space - r.xyz contains the scale for each axis
    var scaled_p = rotated_p * r.xyz;
    
    // Calculate distance from point to sphere center
    // Note: r.w is the global scale/radius of the sphere
    return (length(scaled_p) - r.w);
}

fn sdf_torus(p: vec3f, r: vec2f, quat: vec4f) -> f32
{
    // Apply rotation using quaternion
    var rotated_p = rotate_vector(p, quat);
    
    // Convert to 2D for torus calculation
    var q = vec2f(length(rotated_p.xz) - r.x, rotated_p.y);
    
    // Return the final distance
    return length(q) - r.y;
}

fn sdf_mandelbulb(p: vec3f) -> vec2f
{
  var w = p;
  var m = dot(w, w);

  var dz = 1.0;
  var i = 0;

  for (i = 0; i < 15; i = i + 1)
  {
    dz = 8.0 * pow(sqrt(m), 7.0) * dz + 1.0;
    var r = length(w);
    var b = 8.0 * acos(w.y / r);
    var a = 8.0 * atan2(w.x, w.z);
    w = p + pow(r, 8.0) * vec3f(sin(b) * sin(a), cos(b), sin(b) * cos(a));

    m = dot(w, w);
    if (m > 256.0)
    {
      break;
    }
  }
  var r = 0.25 * log(m) * sqrt(m) / dz;
  return vec2f(r, f32(i) / 16.0);
}

fn sdf_weird_thing(p_: vec3f, s: f32) -> f32
{
  var scale = 1.0;
  var orb = vec4f(1000.0);
  var p = p_;

  for (var i = 0; i < 8; i = i + 1)
  {
    p = -1.0 + 2.0 * modc(0.5 * p + 0.5, vec3f(1.0));

    var r2 = dot(p, p);
    orb = min(orb, vec4f(abs(p), r2));

    var k = s / r2;
    p *= k;
    scale *= k;
  }

  return 0.3 * abs(p.y) / scale;
}