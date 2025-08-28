// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "utils.h"

TEST_CASE("Test IVF Bucket Statistics", "Record") {
    auto metric = knowhere::metric::L2;
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    int64_t nb = 10000, nq = 10;
    int64_t dim = 128;
    int64_t seed = 42;
    int64_t top_k = 100;
    int64_t nprobe = 16;

    knowhere::Json json;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::indexparam::NLIST] = 123;
    json[knowhere::indexparam::NPROBE] = nprobe;
    json[knowhere::indexparam::RECORD_BUCKET_STATS] = true;
    json[knowhere::indexparam::BUCKET_STATS_FILE] = "bucket_stats.csv";
    json[knowhere::indexparam::RETURN_VISITED_BUCKETS] = true;

    SECTION("Simple Train") {
        auto train_ds = GenDataSet(nb, dim, seed);
        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, version)
                       .value();
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.Type() == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT);
        REQUIRE(idx.HasRawData(metric) == knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(
                                              knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, version, json));
    }

    SECTION("Simple Search") {
        auto train_ds = GenDataSet(nb, dim, seed);
        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, version)
                       .value();
        auto build_res = idx.Build(train_ds, json);
        REQUIRE(build_res == knowhere::Status::success);
        REQUIRE(idx.Type() == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT);

        auto search_ds = GenDataSet(nq, dim, seed);
        auto search_res = idx.Search(search_ds, json, nullptr).value();
        auto visited_bucket_ids = search_res->Get<std::vector<int64_t>>(knowhere::meta::VISITED_BUCKET_IDS);

        json[knowhere::indexparam::NPROBE] = nprobe * 2;
        auto search_res_2 = idx.Search(search_ds, json, nullptr).value();
        auto visited_bucket_ids_2 = search_res_2->Get<std::vector<int64_t>>(knowhere::meta::VISITED_BUCKET_IDS);
        for (size_t i = 0; i < nq; i++) {
            for (size_t j = 0; j < nprobe; j++) {
                REQUIRE(visited_bucket_ids[i * nprobe + j] == visited_bucket_ids_2[i * nprobe * 2 + j]);
            }
        }
    }
}
