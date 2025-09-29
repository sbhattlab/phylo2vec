#!/usr/bin/env ruby
# A simple script to find a paper.md or paper.tex file in a directory
# Adapted from https://github.com/openjournals/buffy/blob/6ef14c38260e7ca2607495e749326d427413da79/app/workers/repo_checks_worker.rb#L53
# and https://github.com/openjournals/buffy/blob/main/app/lib/paper_file.rb


require 'find'
require 'open3'

class SimplePaperFile
  attr_accessor :paper_path

  def initialize(path = nil)
    @paper_path = path
  end

  def self.find(search_path)
    paper_path = nil

    if Dir.exist?(search_path)
      Find.find(search_path) do |path|
        if path =~ /\/paper\.tex$|\/paper\.md$/
          paper_path = path
          break
        end
      end
    end

    SimplePaperFile.new(paper_path)
  end

  def count_words
    return nil if @paper_path.nil?

    word_count = Open3.capture3("cat #{@paper_path} | wc -w")[0].to_i
    puts "Wordcount for `#{File.basename(@paper_path)}` is #{word_count}"
    word_count
  end

  def text
    return "" if @paper_path.nil? || @paper_path.empty?
    File.read(@paper_path)
  end
end

# Usage example
if __FILE__ == $0
  search_directory = ARGV[0] || "."

  paper_file = SimplePaperFile.find(search_directory)

  if paper_file.paper_path
    puts "Found paper: #{paper_file.paper_path}"
    paper_file.count_words
  else
    puts "No paper.md or paper.tex file found in #{search_directory}"
    puts "Looking for files matching /paper\.(md|tex)$/"
  end
end