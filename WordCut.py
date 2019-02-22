#! /bin/env python
# -*- coding: utf-8 -*-
"""
分词，去停用词
去停用词可以理解成去噪声
"""
import jieba
import os

stopwords_path = "/stopwords.txt"


class WordCut ( object ) :
	def __init__ ( self, stopwords_path = stopwords_path ) :
		"""
        :stopwords_path: 停用词文件路径

        """
		self.stopwords_path = stopwords_path
	
	def addDictionary ( self, dict_list ) :
		"""
        添加用户自定义字典列表
        """
		map ( lambda x : jieba.load_userdict ( x ), dict_list )
	
	def seg_sentence ( self, sentence, stopwords_path = None ) :
		"""
        对句子进行分词  
        """
		# print "now token sentence..."
		if stopwords_path is None :
			stopwords_path = self.stopwords_path
		
		def stopwordslist ( filepath ) :
			"""
            创建停用词list ,闭包 
            """
			stopwords = [ line.decode ( 'utf-8' ).strip () for line in open ( filepath, 'rb' ).readlines () ]
			return stopwords
		
		sentence_seged = jieba.cut ( sentence.strip () )
		stopwords = stopwordslist ( stopwords_path )  # 这里加载停用词的路径
		outstr = ''  # 返回值是字符串
		for word in sentence_seged :
			if word not in stopwords :
				if word != '\t' :
					outstr += word
					outstr += " "
		return outstr

