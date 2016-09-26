require 'narray'

class KMeansClusterer
  TYPECODE = { double: NArray::DFLOAT, single: NArray::SFLOAT }

  module Utils
    def self.ensure_matrix x, typecode = nil
      if x.is_a?(NMatrix)
        x
      elsif defined?(GSL::Matrix) && x.is_a?(GSL::Matrix)
        x.to_nm
      else
        NMatrix.cast(x, typecode)
      end
    end

    def self.ensure_narray x, typecode = nil
      x.is_a?(NArray) ? x : NArray.cast(x, typecode)
    end
  end
  [ [ 32.2909, -87.8799 ], [ 33.209, -85.7719 ], [ 31.6497, -86.012 ], [ 34.1931, -86.8009 ] ]


  module Scaler
    def self.mean data
      data.mean(1)
    end

    def self.std data
      std = data.stddev(1)
      std[std.eq(0)] = 1.0 # so we don't divide by 0
      std
    end

    def self.scale data, mean = nil, std = nil, typecode = nil
      data = NArray.ref(data)
      mean ||= self.mean(data)
      std ||= self.std(data)
      data = (data - mean) / std
      [NMatrix.ref(data), mean, std]
    end

    def self.row_norms data
      squared_data = NArray.ref(data)**2
      NMatrix.ref(squared_data).sum(0)
    end
  end

  module Distance
    def self.euclidean x, y, yy = nil
      if x.is_a?(NMatrix) && y.is_a?(NMatrix)
        xx = Scaler.row_norms(x)
        yy ||= Scaler.row_norms(y)
        xy = x * y.transpose
        distance = xy * -2
        distance += xx
        distance += yy.transpose
        NMath.sqrt distance
      else
        NMath.sqrt ((x - y)**2).sum(0)
      end
    end
  end


  class Point
    attr_reader :id, :data, :centroid_distances
    attr_accessor :cluster, :label

    def initialize id, data, centroid_distances, label = nil
      @id = id
      @data = data
      @centroid_distances = centroid_distances
      @label = label
    end

    def [] index
      @data[index]
    end

    def to_a
      @data.to_a
    end

    def to_s
      to_a.to_s
    end
    
    
    def dimension
      @data.length
    end

    def centroid_distance
      @centroid_distances[@cluster.id]
    end
  end


  class Cluster
    attr_reader :id, :centroid, :points
    attr_accessor :label

    def initialize id, centroid
      @id = id
      @centroid = centroid
      @points = []
    end

    def << point
      point.cluster = self
      @points << point
    end

    def sorted_points point = @centroid
      point = point.data if point.is_a?(Point)
      point = NArray.cast(point, @centroid.typecode) unless point.is_a?(NArray)
      points_data = NArray.cast(@points.map(&:data))
      distances = Distance.euclidean(points_data, point)
      @points.sort_by.with_index {|p, i| distances[i] }
    end
  end


  DEFAULT_OPTS = { scale_data: false, runs: 10, log: false, init: :kmpp, float_precision: :double, max_iter: 300 }

  def self.run k, data, opts = {}
    opts = DEFAULT_OPTS.merge(opts)

    opts[:k] = k
    typecode = TYPECODE[opts[:float_precision]]

    data = Utils.ensure_matrix data, typecode

    if opts[:scale_data]
      data, mean, std = Scaler.scale(data, nil, nil, typecode)
      opts[:mean] = mean
      opts[:std] = std
    end

    opts[:data] = data
    opts[:row_norms] = Scaler.row_norms(data)

    bestrun = nil

    opts[:runs].times do |i|
      km = new(opts).run

      if opts[:log]
        puts "[#{i + 1}] #{km.iterations} iter\t#{km.runtime.round(2)}s\t#{km.error.round(2)} err"
      end
      
      if bestrun.nil? || (km.error < bestrun.error)
        bestrun = km
      end
    end

    bestrun.finish
  end


  attr_reader :k, :points, :clusters, :centroids, :error, :mean, :std, :iterations, :runtime, :distances, :data


  def initialize opts = {}
    @k = opts[:k]
    @init = opts[:init]
    @labels = opts[:labels] || []
    @row_norms = opts[:row_norms]

    @data = opts[:data]
    @points_count = @data ? @data.shape[1] : 0
    @mean = Utils.ensure_narray(opts[:mean]) if opts[:mean]
    @std = Utils.ensure_narray(opts[:std]) if opts[:std]
    @scale_data = opts[:scale_data]
    @typecode = TYPECODE[opts[:float_precision] || :double]
    @max_iter = opts[:max_iter]
    @max_points_in_a_cluster = (@points_count/@k).to_i

    init_centroids
  end

  def run 
    start_time = Time.now
    @iterations, @runtime = 0, 0
    @cluster_assigns = NArray.int(@points_count)
    min_distances = NArray.new(@typecode, @points_count)
    loop do
      @iterations +=1

      min_distances.fill! Float::INFINITY
      @distances = Distance.euclidean(@centroids, @data, @row_norms)

      @k.times do |cluster_id|
        dist = NArray.ref @distances[true, cluster_id].flatten
        mask = dist < min_distances
        @cluster_assigns[mask] = cluster_id
        min_distances[mask] = dist[mask]
      end

      max_move = 0

      @k.times do |cluster_id|
        centroid = NArray.ref(@centroids[true, cluster_id].flatten)
        point_ids = @cluster_assigns.eq(cluster_id).where
        
        unless point_ids.empty?
          points = @data[true, point_ids]
          newcenter = points.mean(1)
          move = Distance.euclidean(centroid, newcenter)
          max_move = move if move > max_move
          #@max_points_in_a_cluster
          @centroids[true, cluster_id] = newcenter
        end
      end

      break if max_move < 0.001 # i.e., no movement
      break if @iterations >= @max_iter
    end

    @error = (min_distances**2).sum
    @runtime =  Time.now - start_time
    self
  end

  def finish
    @clusters = @k.times.map do |i|
      centroid = NArray.ref @centroids[true, i].flatten
      Cluster.new i, Point.new(-1, centroid, nil, nil)
    end
    
    @points = @points_count.times.map do |i|
      data = NArray.ref @data[true, i].flatten
      point = Point.new(i, data, @distances[i, true], @labels[i])
      cluster = @clusters[@cluster_assigns[i]]
      #if cluster.points.count < @max_points_in_a_cluster
        cluster << point
      #else
      #  unused_points << point
      #end
      point
    end
    
    #@points
    
    
    @clusters.each do |c| 
      c.points.sort_by! &:centroid_distance
    end
    0.upto(8) do
    @clusters.each do |cluster|
      if cluster.points.count > @max_points_in_a_cluster
        original_indexes = []
        @max_points_in_a_cluster.upto((cluster.points.size-1)) do |point_index|
          extra_point = cluster.sorted_points[point_index]
          new_centroids = @centroids.to_a.reject{|x| x == cluster.centroid.data}
          next_clust_predict = predict [extra_point.data], NMatrix.cast(new_centroids, NArray::DFLOAT)
          next_cluster = @clusters[next_clust_predict[0][1][0]]
          if next_cluster.points.count < @max_points_in_a_cluster
            next_cluster << extra_point
            original_indexes << cluster.points.to_a.index{|x| x.id==extra_point.id}
          end
        end
        original_indexes.each {|ind| cluster.points.delete_at(ind)}
      end
      
    end
  end
    
    #@points.each do |point|
    #  next_cluster_id = predict [point.data]
    #  @k.times.map do |id|
    #    if @clusters[next_cluster_id[0][id][0]].points.count < @max_points_in_a_cluster
    #      cluster = @clusters[next_cluster_id[0][id][0]]
    #      cluster << point
    #    end
    #  end
    #end
    #
    ###0.upto(256) do
    #  @clusters.each do |clust|
    #    max = clust.points.count-1
    #    0.upto(max) do |i|
    #      p "loop"
    #      next_cluster_id = predict [clust.points[i].data]
    #      if next_cluster_id[0][0][0] != clust.id.to_i
    #        new_clust = @clusters[next_cluster_id[0][0][0]]
    #        0.upto(max) do |new_i|
    #          p " inner loop 2"
    #          next_clust_predict = predict [new_clust.points[new_i].data]
    #          if next_clust_predict[0][0][0] == clust.id.to_i
    #            p "most inner loop"
    #            temp = clust.points[i]
    #            clust.points.delete_at(i)
    #            new_clust.points << temp
    #            temp = new_clust.points[new_i]
    #            new_clust.points.delete_at(new_i)
    #            clust.points << temp
    #            
    #          #elsif next_clust_predict[0][1][0] == clust.id.to_i
    #          #  p "most inner loop 2"
    #          #  temp = clust.points[i]
    #          #  clust.points.delete_at(i)
    #          #  new_clust.points << temp
    #          #  temp = new_clust.points[new_i]
    #          #  new_clust.points.delete_at(new_i)
    #          #  clust.points << temp
    #          #  break
    #          #elsif next_clust_predict[0][2][0] == clust.id.to_i
    #          #  p "most inner loop 2"
    #          #  temp = clust.points[i]
    #          #  clust.points.delete_at(i)
    #          #  new_clust.points << temp
    #          #  temp = new_clust.points[new_i]
    #          #  new_clust.points.delete_at(new_i)
    #          #  clust.points << temp
    #          #  break
    #          #elsif next_clust_predict[0][3][0] == clust.id.to_i
    #          #  p "most inner loop 2"
    #          #  temp = clust.points[i]
    #          #  clust.points.delete_at(i)
    #          #  new_clust.points << temp
    #          #  temp = new_clust.points[new_i]
    #          #  new_clust.points.delete_at(new_i)
    #          #  clust.points << temp
    #          #  break
    #          end
    #        end
    #        
    #        
    #      end
    #      
    #    end
    #  end
      
      
    #end

    self
  end
  
  def distance loc1, loc2
    rad_per_deg = Math::PI/180  # PI / 180
    rkm = 6371                  # Earth radius in kilometers
    rm = rkm * 1000             # Radius in meters
  
    dlat_rad = (loc2[0]-loc1[0]) * rad_per_deg  # Delta, converted to rad
    dlon_rad = (loc2[1]-loc1[1]) * rad_per_deg
  
    lat1_rad, lon1_rad = loc1.map {|i| i * rad_per_deg }
    lat2_rad, lon2_rad = loc2.map {|i| i * rad_per_deg }
  
    a = Math.sin(dlat_rad/2)**2 + Math.cos(lat1_rad) * Math.cos(lat2_rad) * Math.sin(dlon_rad/2)**2
    c = 2 * Math::atan2(Math::sqrt(a), Math::sqrt(1-a))
  
    rm * c # Delta in meters
  end

  def predict data,centroids
    data = Utils.ensure_matrix data, @typecode
    data, _m, _s = Scaler.scale(data, @mean, @std, @typecode) if @scale_data
    distances = Distance.euclidean(centroids, data)
    data.shape[1].times.map do |i|
      distances[i, true].sort_index.to_a # index of closest cluster
    end
  end

  def sorted_clusters point = origin
    point = point.data if point.is_a?(Point)
    point = NArray.cast(point, @typecode) unless point.is_a?(NArray)
    distances = Distance.euclidean(NArray.ref(@centroids), point)
    @clusters.sort_by.with_index {|c, i| distances[i] }
  end

  def silhouette
    return 1.0 if @k < 2

    # calculate all point-to-point distances at once
    # uses more memory, but much faster
    point_distances = Distance.euclidean @data, @data

    scores = @points.map do |point|
      dissimilarities = @clusters.map do |cluster|  
        dissimilarity(point.id, cluster.id, point_distances)
      end
      a = dissimilarities[point.cluster.id]
      # set to Infinity so we can pick next closest via min()
      dissimilarities[point.cluster.id] = Float::INFINITY
      b = dissimilarities.min

      (b - a) / [a,b].max
    end

    scores.reduce(:+) / scores.length # mean score for all points
  end

  def inspect
    %{#<#{self.class.name} k:#{@k} iterations:#{@iterations} error:#{@error} runtime:#{@runtime}>}
  end

  private

    def dissimilarity point_id, cluster_id, point_distances
      cluster_point_ids = @cluster_assigns.eq(cluster_id).where
      cluster_point_distances = point_distances[cluster_point_ids, point_id]
      cluster_point_distances.mean
    end

    def init_centroids
      case @init
      when :random
        random_centroid_init
      when Array
        custom_centroid_init
      else
        kmpp_centroid_init
      end
    end

    # k-means++
    def kmpp_centroid_init
      centroid_ids = []
      pick = rand(@points_count)
      centroid_ids << pick

      while centroid_ids.length < @k
        centroids = @data[true, centroid_ids]
        distances = Distance.euclidean(centroids, @data, @row_norms)
        
        # squared distances of each point to the nearest centroid
        d2 = NArray.ref(distances.min(1).flatten)**2

        probs = d2 / d2.sum
        cumprobs = probs.cumsum
        r = rand
        pick = (cumprobs >= r).where[0]
        centroid_ids << pick
      end

      @centroids = @data[true, centroid_ids]
    end

    def custom_centroid_init
      @centroids = NMatrix.cast @init, @typecode
      @k = @init.length
    end

    def random_centroid_init
      @centroids = @data[true, pick_k_random_indexes]
    end

    def pick_k_random_indexes
      @points_count.times.to_a.sample @k
    end

    def origin
      Array.new(@points[0].dimension, 0) 
    end
end
